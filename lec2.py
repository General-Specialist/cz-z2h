# To run this in Google Colab, install Gemmi first:
# !pip install gemmi

import os
import random
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import gemmi

# ==============================================================================
# THE CRYOZETA DOWNSCALING PARADIGM (Real PDB Training & Colab Note)
# ==============================================================================
# CryoZeta is trained on a massive scale: 48 EM-Pairformer blocks, 1024-dim
# channels, and thousands of full-resolution (256^3) experimental maps.
#
# This file implements the EXACT SAME volumetric learning pipeline, but designed
# to run in a few minutes in a single Google Colab notebook cell (using T4 GPU):
# - Data Source: Sourced directly from the RCSB Protein Data Bank (PDB).
# - Bounding Box: Extracting local 16.0 Å patches from real folded proteins.
# - Grid Dimensions: Downscaled from 256^3 -> 32^3 (retaining 3D spatial properties).
# - Voxel Resolution: 0.5 Å grid spacing (representing a 16.0 Å box).
# - Model Capacity: 16-feature initial channels (easily expandable to 32 or 64).
# - Landmark Bottleneck: Extracts M=16 support points to act as geometric anchors.
#
# The U-Net is trained on multiple real proteins (e.g. Ubiquitin, Rubredoxin, BPTI)
# and validated on a completely unseen plant seed protein (Crambin, 1CRN).
# ==============================================================================


# Standard amino acid residues to filter out water, ions, and ligands
PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}


# ==============================================================================
# SECTION 1: PDB DATA PIPELINE (Downloading, Caching, and Cropping)
# ==============================================================================

def download_pdb_cif(pdb_id: str, save_dir: str = "./pdb_data") -> str:
    """
    Downloads and caches a .cif structure file directly from the RCSB PDB server.
    """
    pdb_id = pdb_id.lower()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{pdb_id}.cif")
    if not os.path.exists(filepath):
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        print(f"Downloading {pdb_id.upper()} from RCSB PDB...")
        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"Failed to download {pdb_id.upper()}: {e}")
            raise e
    return filepath


def load_and_crop_pdb(filepath: str, box_size: float = 16.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loads a macromolecular structure, selects a random anchor residue, and crops a
    local spatial sub-volume of size box_size around it.
    Returns:
        all_coords_centered: [N_all, 3] shifted coordinates in [0, box_size]
        carbon_coords_centered: [N_carbons, 3] shifted coordinates in [0, box_size]
    """
    structure = gemmi.read_structure(filepath)

    all_atoms = []
    carbon_atoms = []

    # Flatten structure and keep only standard protein atoms
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.name.strip().upper()
                if res_name not in PROTEIN_RESIDUES:
                    continue
                for atom in residue:
                    pos = [atom.pos.x, atom.pos.y, atom.pos.z]
                    elem = atom.element.name.strip().upper()

                    all_atoms.append(pos)
                    if elem == "C":
                        carbon_atoms.append(pos)

    if not all_atoms:
        raise ValueError(f"No valid protein residues found in {filepath}")

    all_coords = torch.tensor(all_atoms, dtype=torch.float32)
    carbon_coords = torch.tensor(carbon_atoms, dtype=torch.float32)

    # Select a random atom as the local crop anchor
    num_atoms = all_coords.shape[0]
    random_idx = torch.randint(0, num_atoms, (1,)).item()
    center_atom = all_coords[random_idx]

    # Keep coordinates falling inside the local box bounds around center_atom
    half_box = box_size / 2.0

    all_mask = torch.all((all_coords >= center_atom - half_box) & (all_coords <= center_atom + half_box), dim=-1)
    cropped_all = all_coords[all_mask]

    carbon_mask = torch.all((carbon_coords >= center_atom - half_box) & (carbon_coords <= center_atom + half_box), dim=-1)
    cropped_carbon = carbon_coords[carbon_mask]

    # Shift coordinates so that center_atom is centered exactly at [box_size/2, box_size/2, box_size/2]
    # This maps the cropped region exactly into the [0, box_size]^3 voxel space.
    cropped_all_centered = cropped_all - center_atom + half_box
    cropped_carbon_centered = cropped_carbon - center_atom + half_box

    return cropped_all_centered, cropped_carbon_centered


def coords_to_density(coords: torch.Tensor, box_size: float = 16.0, grid_size: int = 32, sigma: float = 0.8) -> torch.Tensor:
    """
    Vectorized and differentiable 3D density rasterization directly in memory.
    Assumes coords are pre-mapped into the physical space of the box: [0, box_size]^3.
    """
    if coords.shape[0] == 0:
        return torch.zeros((grid_size, grid_size, grid_size), device=coords.device)

    device = coords.device
    # Generate the 3D grid ticks
    ticks = torch.linspace(0.0, box_size, grid_size, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(ticks, ticks, ticks, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1) # Shape: [G, G, G, 3]

    # Compute squared distances between every grid voxel and every atom coordinate
    grid_expanded = grid.unsqueeze(-2) # Shape: [G, G, G, 1, 3]
    coords_expanded = coords[None, None, None, :, :] # Shape: [1, 1, 1, N, 3]

    sq_distances = torch.sum((grid_expanded - coords_expanded) ** 2, dim=-1) # Shape: [G, G, G, N]

    # Apply Gaussian radial basis function to construct the continuous map
    atom_densities = torch.exp(-sq_distances / (2 * (sigma ** 2)))
    density_map = atom_densities.sum(dim=-1) # Shape: [G, G, G]
    return density_map


def generate_cryo_em_sample(filepaths: list[str], box_size: float = 16.0, grid_size: int = 32, noise_level: float = 0.05) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Crops a local spatial sub-volume from a random protein and rasterizes:
    - Input: all atoms, wider blur (lower resolution), with added Gaussian noise.
    - Target: carbon atoms only, sharp blur, clean.
    """
    filepath = random.choice(filepaths)

    # Retry cropping if we get a volume with too few carbon atoms
    all_coords, carbon_coords = torch.zeros(0), torch.zeros(0)
    for _ in range(15):
        try:
            all_coords, carbon_coords = load_and_crop_pdb(filepath, box_size=box_size)
            if carbon_coords.shape[0] >= 5:
                break
        except Exception:
            continue

    if carbon_coords.shape[0] == 0:
        raise ValueError(f"Could not extract a valid crop from PDB structural dataset.")

    # Input map: all atoms, wider blur (simulating low-resolution cryo-EM map)
    input_density = coords_to_density(all_coords, box_size=box_size, grid_size=grid_size, sigma=1.2)
    noise = torch.randn_like(input_density) * noise_level
    input_density = F.relu(input_density + noise) # clamp negative densities to 0

    # Target map: carbons only, sharp blur (simulating ground-truth carbon positions)
    target_density = coords_to_density(carbon_coords, box_size=box_size, grid_size=grid_size, sigma=0.6)

    return input_density, target_density, carbon_coords


# ==============================================================================
# SECTION 2: THE 3D U-NET ARCHITECTURE (Volumetric Segmenter)
# ==============================================================================

class DoubleConv3D(nn.Module):
    """
    Fundamental 3D building block: (Conv3D -> BatchNorm3D -> ReLU) * 2.
    Processes volumetric feature maps while maintaining spatial dimensions.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    """
    Fully batched and parameterized 3D U-Net Segmenter.
    Downsamples the 3D density grid using MaxPool3D to extract multi-scale spatial features,
    then upsamples using ConvTranspose3D while concatenating high-resolution skip features
    directly from the encoder path to preserve exact atomic boundaries.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_features: int = 8) -> None:
        super().__init__()
        F_dim = init_features

        # --- Encoder (Downsampling Path) ---
        self.down1 = DoubleConv3D(in_channels, F_dim)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # Halves resolution: 32^3 -> 16^3

        self.down2 = DoubleConv3D(F_dim, F_dim * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # Halves resolution: 16^3 -> 8^3

        # --- Bottleneck ---
        self.bottleneck = DoubleConv3D(F_dim * 2, F_dim * 4)

        # --- Decoder (Upsampling Path) ---
        self.up1 = nn.ConvTranspose3d(F_dim * 4, F_dim * 2, kernel_size=2, stride=2)  # Upsamples: 8^3 -> 16^3
        self.conv_up1 = DoubleConv3D(F_dim * 4, F_dim * 2)

        self.up2 = nn.ConvTranspose3d(F_dim * 2, F_dim, kernel_size=2, stride=2)  # Upsamples: 16^3 -> 32^3
        self.conv_up2 = DoubleConv3D(F_dim * 2, F_dim)

        self.out_conv = nn.Conv3d(F_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder ---
        x1 = self.down1(x)              # Shape: [B, F_dim, 32, 32, 32]
        p1 = self.pool1(x1)             # Shape: [B, F_dim, 16, 16, 16]

        x2 = self.down2(p1)             # Shape: [B, 2*F_dim, 16, 16, 16]
        p2 = self.pool2(x2)             # Shape: [B, 2*F_dim, 8, 8, 8]

        # --- Bottleneck ---
        b = self.bottleneck(p2)         # Shape: [B, 4*F_dim, 8, 8, 8]

        # --- Decoder ---
        u1 = self.up1(b)                # Shape: [B, 2*F_dim, 16, 16, 16]
        c1 = torch.cat([u1, x2], dim=1) # Skip connection
        x3 = self.conv_up1(c1)          # Shape: [B, 2*F_dim, 16, 16, 16]

        u2 = self.up2(x3)               # Shape: [B, F_dim, 32, 32, 32]
        c2 = torch.cat([u2, x1], dim=1) # Skip connection
        x4 = self.conv_up2(c2)          # Shape: [B, F_dim, 32, 32, 32]

        out = self.out_conv(x4)
        return out


# ==============================================================================
# SECTION 3: 3D NON-MAXIMUM SUPPRESSION (Peak-Finders)
# ==============================================================================

class BatchedPeakFinder3D(nn.Module):
    """
    Bridges the dense 3D voxel grid with sequence representations by extracting
    a fixed number (M) of high-confidence spatial landmarks (support points).
    """
    def __init__(self, threshold: float = 0.15, kernel_size: int = 3, max_peaks: int = 16, box_size: float = 16.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.max_peaks = max_peaks
        self.box_size = box_size

    def forward(self, density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, X, Y, Z = density.shape
        assert C == 1, "Peak finder expects single-channel density map"

        device = density.device
        padding = self.kernel_size // 2

        # 1. Isolate local maxima using a 3D max-pooling filter
        max_pooled = F.max_pool3d(density, kernel_size=self.kernel_size, stride=1, padding=padding)
        is_peak_mask = (density == max_pooled) & (density > self.threshold)

        M = self.max_peaks
        out_coords = torch.zeros((B, M, 3), dtype=torch.float32, device=device)
        out_values = torch.zeros((B, M), dtype=torch.float32, device=device)
        out_mask = torch.zeros((B, M), dtype=torch.bool, device=device)

        ticks = torch.linspace(0.0, self.box_size, X, device=device)

        for b in range(B):
            sample_peaks = is_peak_mask[b, 0]
            indices = torch.nonzero(sample_peaks)

            if indices.shape[0] == 0:
                continue

            values = density[b, 0][sample_peaks]
            sorted_indices = torch.argsort(values, descending=True)
            indices = indices[sorted_indices]
            values = values[sorted_indices]

            num_to_copy = min(indices.shape[0], M)

            if num_to_copy > 0:
                selected_indices = indices[:num_to_copy]

                # Map grid indices to physical Angstrom coordinates in the box
                phys_x = ticks[selected_indices[:, 0]]
                phys_y = ticks[selected_indices[:, 1]]
                phys_z = ticks[selected_indices[:, 2]]

                out_coords[b, :num_to_copy] = torch.stack([phys_x, phys_y, phys_z], dim=-1)
                out_values[b, :num_to_copy] = values[:num_to_copy]
                out_mask[b, :num_to_copy] = True

        return out_coords, out_values, out_mask


# ==============================================================================
# SECTION 4: GENERALIZED REAL-DATA TRAINING & COORDINATE DECODING
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    # Select execution device (Apple Silicon MPS, NVIDIA CUDA, or CPU) early to accelerate pre-caching
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # 1. Download structural files from RCSB PDB
    print("===========================================================================")
    print(" PREPARING PDB MOLECULAR STRUCTURAL REPOSITORY ")
    print("===========================================================================")

    # Training set proteins (Ubiquitin, Rubredoxin, BPTI, Signaling Domains)
    train_ids = ["1ubq", "1a8o", "1bpi", "1cjg", "1eyy"]
    # Unseen testing protein (Crambin - extremely clean plant-seed lipid transfer protein)
    test_id = "1crn"

    train_files = []
    for pid in train_ids:
        try:
            path = download_pdb_cif(pid)
            train_files.append(path)
        except Exception:
            print(f"Skipping {pid} due to download error.")

    test_file = download_pdb_cif(test_id)

    print(f"\nSuccessfully cached {len(train_files)} training structures and 1 unseen test structure.")

    # 2. Load full PDB coordinates once into memory to make dynamic cropping ultra-fast
    print("\nLoading PDB structures into memory once...")
    train_structures = []
    for filepath in train_files:
        try:
            structure = gemmi.read_structure(filepath)
            all_atoms = []
            carbon_atoms = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        res_name = residue.name.strip().upper()
                        if res_name not in PROTEIN_RESIDUES:
                            continue
                        for atom in residue:
                            pos = [atom.pos.x, atom.pos.y, atom.pos.z]
                            all_atoms.append(pos)
                            if atom.element.name.strip().upper() == "C":
                                carbon_atoms.append(pos)
            all_coords = torch.tensor(all_atoms, dtype=torch.float32).to(device)
            carbon_coords = torch.tensor(carbon_atoms, dtype=torch.float32).to(device)
            train_structures.append((all_coords, carbon_coords))
            print(f"  Loaded {os.path.basename(filepath)} | Atoms: {len(all_atoms)} | Carbons: {len(carbon_atoms)}")
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

    if not train_structures:
        raise ValueError("Could not parse any protein structures into memory.")

    # Dynamic dynamic-cropping function
    def crop_and_rasterize_dynamic(structures: list, box_size: float = 16.0, grid_size: int = 32, noise_level: float = 0.04) -> tuple[torch.Tensor, torch.Tensor]:
        # Pick a random structure
        all_coords, carbon_coords = random.choice(structures)

        # Crop retry block to ensure we always get a valid region with carbon atoms
        for _ in range(15):
            num_atoms = all_coords.shape[0]
            random_idx = torch.randint(0, num_atoms, (1,)).item()
            center_atom = all_coords[random_idx]

            half_box = box_size / 2.0
            carbon_mask = torch.all((carbon_coords >= center_atom - half_box) & (carbon_coords <= center_atom + half_box), dim=-1)
            cropped_carbon = carbon_coords[carbon_mask]

            if cropped_carbon.shape[0] >= 5:
                all_mask = torch.all((all_coords >= center_atom - half_box) & (all_coords <= center_atom + half_box), dim=-1)
                cropped_all = all_coords[all_mask]
                break

        # Shift coordinates to align inside the [0, box_size]^3 space
        cropped_all_centered = cropped_all - center_atom + half_box
        cropped_carbon_centered = cropped_carbon - center_atom + half_box

        # Rasterize inputs and targets on the fly
        input_density = coords_to_density(cropped_all_centered, box_size=box_size, grid_size=grid_size, sigma=1.2)
        noise = torch.randn_like(input_density) * noise_level
        input_density = F.relu(input_density + noise)

        target_density = coords_to_density(cropped_carbon_centered, box_size=box_size, grid_size=grid_size, sigma=0.6)

        return input_density, target_density

    # Pre-cache training and validation datasets at startup to prevent CPU bottlenecks
    print("\nPre-caching 160 training crops from PDB structures...")
    train_dataset = []
    for _ in range(160):
        train_input, train_target = crop_and_rasterize_dynamic(train_structures, box_size=16.0, grid_size=32, noise_level=0.04)
        train_dataset.append((train_input, train_target))

    print("Pre-caching 40 validation crops...")
    val_dataset = []
    for _ in range(40):
        val_input, val_target = crop_and_rasterize_dynamic(train_structures, box_size=16.0, grid_size=32, noise_level=0.04)
        val_dataset.append((val_input, val_target))

    # Initialize U-Net, optimizer, and scheduler
    model = UNet3D(in_channels=1, out_channels=1, init_features=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)
    criterion = nn.MSELoss()

    # Select execution device
    model.to(device)
    print(f"Running on computational device: {device}")

    peak_finder = BatchedPeakFinder3D(threshold=0.15, max_peaks=128, box_size=16.0)
    peak_finder.to(device)

    print("\n" + "="*75)
    print(" RUNNING REAL PDB U-NET GENERALIZATION TRAINING ")
    print("="*75)

    epochs = 40
    batch_size = 8

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        # Shuffle indices of the pre-cached training dataset
        shuffled_indices = torch.randperm(len(train_dataset))

        for i in range(0, len(train_dataset), batch_size):
            batch_indices = shuffled_indices[i : i + batch_size]
            batch_samples = [train_dataset[idx] for idx in batch_indices]

            inputs = torch.stack([sample[0] for sample in batch_samples]).unsqueeze(1).to(device)  # [B, 1, 32, 32, 32]
            targets = torch.stack([sample[1] for sample in batch_samples]).unsqueeze(1).to(device) # [B, 1, 32, 32, 32]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_samples)

        train_loss /= len(train_dataset)

        # Validation evaluation on the stable, pre-cached set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_input, val_target in val_dataset:
                val_input_tensor = val_input.unsqueeze(0).unsqueeze(0).to(device)
                val_target_tensor = val_target.unsqueeze(0).unsqueeze(0).to(device)
                val_out = model(val_input_tensor)
                loss = criterion(val_out, val_target_tensor)
                val_loss += loss.item()
            val_loss /= len(val_dataset)

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:02d}/{epochs} | LR: {current_lr:.6f} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Evaluation on a completely unseen real protein (Crambin 1CRN)
    print("\n" + "="*75)
    print(" EVALUATION ON UNSEEN PDB STRUCTURE: CRAMBIN (1CRN) ")
    print("="*75)

    test_input, test_target, test_gt_coords = generate_cryo_em_sample([test_file], box_size=16.0, grid_size=32, noise_level=0.04)

    model.eval()
    with torch.no_grad():
        test_in_batch = test_input.unsqueeze(0).unsqueeze(0).to(device)
        pred_density = F.relu(model(test_in_batch))
        pred_coords, pred_vals, pred_mask = peak_finder(pred_density)

    pred_coords = pred_coords[0].cpu()
    pred_mask = pred_mask[0].cpu()
    pred_vals = pred_vals[0].cpu()

    num_pred_peaks = pred_mask.sum().item()
    num_gt_peaks = len(test_gt_coords)

    print(f"Unseen test Crambin crop has {num_gt_peaks} Ground Truth Carbon atoms.")
    print(f"U-Net + Peak Finder resolved {num_pred_peaks} peaks.\n")

    print("Matched Recovered Coordinates vs Ground Truth (Within 1.0 A Tolerance):")
    matched_count = 0

    for gt_idx, gt_c in enumerate(test_gt_coords):
        if num_pred_peaks > 0:
            distances = torch.norm(pred_coords[:num_pred_peaks] - gt_c, dim=-1)
            min_dist, min_idx = torch.min(distances, dim=0)
            if min_dist.item() <= 1.0:
                matched_count += 1
                pred_c = pred_coords[min_idx]
                val = pred_vals[min_idx]
                print(f"  GT Carbon {gt_idx+1:02d}: [{gt_c[0]:.2f}, {gt_c[1]:.2f}, {gt_c[2]:.2f}] -> RESOLVED (Error: {min_dist.item():.3f} Å, Conf: {val:.3f})")
            else:
                print(f"  GT Carbon {gt_idx+1:02d}: [{gt_c[0]:.2f}, {gt_c[1]:.2f}, {gt_c[2]:.2f}] -> MISSED (Closest prediction is {min_dist.item():.3f} Å away)")
        else:
            print(f"  GT Carbon {gt_idx+1:02d}: [{gt_c[0]:.2f}, {gt_c[1]:.2f}, {gt_c[2]:.2f}] -> MISSED (No peaks predicted)")

    accuracy = (matched_count / num_gt_peaks) * 100
    print(f"\nCoordinate Recovery Accuracy: {accuracy:.1f}% ({matched_count}/{num_gt_peaks} carbons resolved within 1.0 Å)")
    print("="*75)
