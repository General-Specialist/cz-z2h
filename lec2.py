import os
import random
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import gemmi

# ==============================================================================
# CONSTANTS & CONFIGURATIONS
# ==============================================================================

# Standard amino acid residues to filter out water, ions, and ligands
PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}

# Directories and Dataset Paths
DEFAULT_SAVE_DIR = "./pdb_data"
TRAIN_PDB_IDS = ["1ubq", "1a8o", "1bpi", "1cjg", "1eyy"]
TEST_PDB_ID = "1crn"

# Volumetric & Rasterization Grid Configuration
DEFAULT_BOX_SIZE = 16.0
DEFAULT_GRID_SIZE = 32
TRAIN_GRID_SIZE = 64

# Gaussian Rasterization Sigmas
DEFAULT_SIGMA = 0.8
INPUT_SIGMA = 1.2
TARGET_SIGMA = 0.6

# Rasterization Chunking (keeps GPU memory footprint under control)
RASTER_CHUNK_SIZE = 128

# Noise Levels
DEFAULT_NOISE_LEVEL = 0.05
TRAIN_NOISE_LEVEL = 0.04

# Carbon Target Detection Radius
DEFAULT_RADIUS = 0.8

# Training & Optimization Hyperparameters
UNET_INIT_FEATURES = 32
LEARNING_RATE = 0.001
SCHEDULER_T_MAX = 60
SCHEDULER_ETA_MIN = 1e-5
NUM_EPOCHS = 60
BATCH_SIZE = 8

# Dataset Pre-caching Configuration
NUM_TRAIN_CROPS = 640
NUM_VAL_CROPS = 120
NUM_TEST_CROPS = 20

# 3D Peak Finding & Post-Processing (Mean-Shift)
DEFAULT_PEAK_THRESHOLD = 0.30
DEFAULT_PEAK_BANDWIDTH = 1.0
DEFAULT_MAX_PEAKS = 128
DEFAULT_PEAK_ITERATIONS = 5
CLASH_LIMIT = 0.6

# Coordinate Metric Evaluation
MATCHING_RADIUS = 1.0


# ==============================================================================
# SECTION 1: UTILITIES, PIPELINES & DATA AUGMENTATION
# ==============================================================================

def download_pdb_cif(pdb_id, save_dir=DEFAULT_SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/{pdb_id}.cif"
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.cif", path)
    return path


def load_and_crop_pdb(filepath: str, box_size: float = DEFAULT_BOX_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
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


def coords_to_density(coords: torch.Tensor, box_size: float = DEFAULT_BOX_SIZE, grid_size: int = DEFAULT_GRID_SIZE, sigma: float = DEFAULT_SIGMA) -> torch.Tensor:
    """
    Vectorized, differentiable, and chunked 3D density rasterization.
    Processes coordinates in chunks to limit GPU memory footprint and prevent CUDA OOM.
    """
    if coords.shape[0] == 0:
        return torch.zeros((grid_size, grid_size, grid_size), device=coords.device)

    device = coords.device
    # Generate the 3D grid ticks
    ticks = torch.linspace(0.0, box_size, grid_size, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(ticks, ticks, ticks, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1) # Shape: [G, G, G, 3]
    g_flat = grid.view(-1, 3) # Shape: [G^3, 3]

    g2 = torch.sum(g_flat ** 2, dim=-1, keepdim=True) # Shape: [G^3, 1]
    density_flat = torch.zeros(g_flat.shape[0], device=device)

    # Process atoms in chunks to cap GPU memory footprint
    chunk_size = RASTER_CHUNK_SIZE
    for i in range(0, coords.shape[0], chunk_size):
        c_chunk = coords[i : i + chunk_size]
        c2_chunk = torch.sum(c_chunk ** 2, dim=-1, keepdim=True).t() # Shape: [1, chunk]

        sq_dists_chunk = g2 + c2_chunk - 2.0 * torch.matmul(g_flat, c_chunk.t())
        sq_dists_chunk = torch.clamp(sq_dists_chunk, min=0.0)

        atom_densities_chunk = torch.exp(-sq_dists_chunk / (2 * (sigma ** 2)))
        density_flat += atom_densities_chunk.sum(dim=-1)

    return density_flat.view(grid_size, grid_size, grid_size)


def coords_to_binary_grid(coords: torch.Tensor, box_size: float = DEFAULT_BOX_SIZE, grid_size: int = DEFAULT_GRID_SIZE, radius: float = DEFAULT_RADIUS) -> torch.Tensor:
    """
    Vectorized, differentiable, and chunked 3D binary grid rasterizer.
    Processes coordinates in chunks to limit GPU memory footprint and prevent CUDA OOM.
    """
    if coords.shape[0] == 0:
        return torch.zeros((grid_size, grid_size, grid_size), device=coords.device)

    device = coords.device
    # Generate the 3D grid ticks
    ticks = torch.linspace(0.0, box_size, grid_size, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(ticks, ticks, ticks, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1) # Shape: [G, G, G, 3]
    g_flat = grid.view(-1, 3) # Shape: [G^3, 3]

    g2 = torch.sum(g_flat ** 2, dim=-1, keepdim=True) # Shape: [G^3, 1]
    min_dists_flat = torch.full((g_flat.shape[0],), float('inf'), device=device)

    # Process atoms in chunks to cap GPU memory footprint
    chunk_size = RASTER_CHUNK_SIZE
    for i in range(0, coords.shape[0], chunk_size):
        c_chunk = coords[i : i + chunk_size]
        c2_chunk = torch.sum(c_chunk ** 2, dim=-1, keepdim=True).t() # Shape: [1, chunk]

        sq_dists_chunk = g2 + c2_chunk - 2.0 * torch.matmul(g_flat, c_chunk.t())
        sq_dists_chunk = torch.clamp(sq_dists_chunk, min=0.0)
        dists_chunk = sq_dists_chunk.sqrt()

        chunk_min, _ = torch.min(dists_chunk, dim=-1)
        min_dists_flat = torch.min(min_dists_flat, chunk_min)

    binary_grid_flat = (min_dists_flat <= radius).float()
    return binary_grid_flat.view(grid_size, grid_size, grid_size)


def augment_batch_3d(inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies boundary-preserving random 3D rotations and flips to the batch.
    Inputs and targets shapes: [B, 1, H, W, D].
    """
    B = inputs.shape[0]
    augmented_inputs = []
    augmented_targets = []

    for b in range(B):
        x = inputs[b]
        y = targets[b]

        # 1. Random Flips (Reflections) - boundary-preserving
        for dim in (-3, -2, -1):
            if random.random() > 0.5:
                x = torch.flip(x, dims=[dim])
                y = torch.flip(y, dims=[dim])

        # 2. Random 90-degree Rotations - boundary-preserving
        for plane in [(-3, -2), (-2, -1), (-3, -1)]:
            k = random.randint(0, 3)
            if k > 0:
                x = torch.rot90(x, k, dims=plane)
                y = torch.rot90(y, k, dims=plane)

        augmented_inputs.append(x)
        augmented_targets.append(y)

    return torch.stack(augmented_inputs), torch.stack(augmented_targets)


class BCEDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction='mean')

        # Soft Dice Loss
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)

        intersection = torch.sum(pred_flat * target_flat, dim=-1)
        union = torch.sum(pred_flat, dim=-1) + torch.sum(target_flat, dim=-1)

        dice = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)
        return bce + dice.mean()


def generate_cryo_em_sample(filepaths: list[str], box_size: float = DEFAULT_BOX_SIZE, grid_size: int = DEFAULT_GRID_SIZE, noise_level: float = DEFAULT_NOISE_LEVEL) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Crops a local spatial sub-volume from a random protein and rasterizes:
    - Input: all atoms, wider blur (lower resolution), with added Gaussian noise.
    - Target: carbon atoms only, sharp blur, clean.
    """
    filepath = random.choice(filepaths)
    all_coords, carbon_coords = load_and_crop_pdb(filepath, box_size=box_size)

    # Input map: all atoms, wider blur (simulating low-resolution cryo-EM map)
    input_density = coords_to_density(all_coords, box_size=box_size, grid_size=grid_size, sigma=INPUT_SIGMA)
    noise = torch.randn_like(input_density) * noise_level
    input_density = F.relu(input_density + noise) # clamp negative densities to 0

    # Target map: carbons only, sharp blur (simulating ground-truth carbon positions)
    target_density = coords_to_density(carbon_coords, box_size=box_size, grid_size=grid_size, sigma=TARGET_SIGMA)

    return input_density, target_density, carbon_coords


# ==============================================================================
# SECTION 2: THE 3D U-NET ARCHITECTURE (Volumetric Segmenter)
# ==============================================================================

class ChannelAttention3D(nn.Module):
    """
    3D Squeeze-and-Excitation Channel Attention module.
    """
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[0], x.shape[1]
        weights = self.fc(x).view(B, C, 1, 1, 1)
        return x * weights


class SpatialAttention3D(nn.Module):
    """
    3D Spatial Attention module using average and max pooling descriptors.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        combined = torch.cat([max_out, mean_out], dim=1)
        weights = torch.sigmoid(self.conv(combined))
        return x * weights


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
    Fully batched and parameterized 3D U-Net Segmenter with Channel & Spatial Attention.
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
        self.bottleneck_att = ChannelAttention3D(F_dim * 4)

        # --- Decoder (Upsampling Path) ---
        self.up1 = nn.ConvTranspose3d(F_dim * 4, F_dim * 2, kernel_size=2, stride=2)  # Upsamples: 8^3 -> 16^3
        self.conv_up1 = DoubleConv3D(F_dim * 4, F_dim * 2)
        self.att1 = SpatialAttention3D()

        self.up2 = nn.ConvTranspose3d(F_dim * 2, F_dim, kernel_size=2, stride=2)  # Upsamples: 16^3 -> 32^3
        self.conv_up2 = DoubleConv3D(F_dim * 2, F_dim)
        self.att2 = SpatialAttention3D()

        self.out_conv = nn.Conv3d(F_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder ---
        x1 = self.down1(x)              # Shape: [B, F_dim, 32, 32, 32]
        p1 = self.pool1(x1)             # Shape: [B, F_dim, 16, 16, 16]

        x2 = self.down2(p1)             # Shape: [B, 2*F_dim, 16, 16, 16]
        p2 = self.pool2(x2)             # Shape: [B, 2*F_dim, 8, 8, 8]

        # --- Bottleneck ---
        b = self.bottleneck(p2)         # Shape: [B, 4*F_dim, 8, 8, 8]
        b = self.bottleneck_att(b)

        # --- Decoder ---
        u1 = self.up1(b)                # Shape: [B, 2*F_dim, 16, 16, 16]
        c1 = torch.cat([u1, x2], dim=1) # Skip connection
        x3 = self.conv_up1(c1)          # Shape: [B, 2*F_dim, 16, 16, 16]
        x3 = self.att1(x3)

        u2 = self.up2(x3)               # Shape: [B, F_dim, 32, 32, 32]
        c2 = torch.cat([u2, x1], dim=1) # Skip connection
        x4 = self.conv_up2(c2)          # Shape: [B, F_dim, 32, 32, 32]
        x4 = self.att2(x4)

        out = self.out_conv(x4)
        return torch.sigmoid(out)


# ==============================================================================
# SECTION 3: 3D NON-MAXIMUM SUPPRESSION (Peak-Finders)
# ==============================================================================

class BatchedMeanShiftPeakFinder3D(nn.Module):
    """
    Continuous 3D Mean-Shift Clustering Peak Finder natively in PyTorch on GPU.
    Seek mode centers in real continuous space with sub-voxel precision.
    """
    def __init__(self, threshold: float = DEFAULT_PEAK_THRESHOLD, bandwidth: float = DEFAULT_PEAK_BANDWIDTH, max_peaks: int = DEFAULT_MAX_PEAKS, box_size: float = DEFAULT_BOX_SIZE, iterations: int = DEFAULT_PEAK_ITERATIONS) -> None:
        super().__init__()
        self.threshold = threshold
        self.bandwidth = bandwidth
        self.max_peaks = max_peaks
        self.box_size = box_size
        self.iterations = iterations

    def forward(self, density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, X, Y, Z = density.shape
        device = density.device

        M = self.max_peaks
        out_coords = torch.zeros((B, M, 3), dtype=torch.float32, device=device)
        out_values = torch.zeros((B, M), dtype=torch.float32, device=device)
        out_mask = torch.zeros((B, M), dtype=torch.bool, device=device)

        ticks = torch.linspace(0.0, self.box_size, X, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(ticks, ticks, ticks, indexing='ij')
        grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1) # Shape: [X, Y, Z, 3]
        flat_grid = grid_coords.view(-1, 3) # Shape: [X*Y*Z, 3]
        spacing = self.box_size / (X - 1)

        # 3D MaxPool filter to select high-quality starting seeds
        max_pooled = F.max_pool3d(density, kernel_size=3, stride=1, padding=1)
        is_peak_mask = (density == max_pooled) & (density > self.threshold)

        for b in range(B):
            sample_density = density[b, 0] # [X, Y, Z]
            sample_peaks = is_peak_mask[b, 0] # [X, Y, Z]

            # 1. Extract active coordinates and weights for distance computations
            active_mask = sample_density > self.threshold
            active_coords = flat_grid[active_mask.view(-1)] # [A, 3]
            active_weights = sample_density[active_mask] # [A]

            # 2. Extract local max-pooling peaks as starting seeds
            seeds = flat_grid[sample_peaks.view(-1)] # [S, 3]
            seed_probs = sample_density[sample_peaks]
            sorted_idx = torch.argsort(seed_probs, descending=True)
            seeds = seeds[sorted_idx[:M]]

            # 3. Iterative Mean-Shift seeks in continuous space
            for _ in range(self.iterations):
                s2 = torch.sum(seeds ** 2, dim=-1, keepdim=True) # [S, 1]
                a2 = torch.sum(active_coords ** 2, dim=-1, keepdim=True).t() # [1, A]
                sq_dists = s2 + a2 - 2.0 * torch.matmul(seeds, active_coords.t()) # [S, A]
                sq_dists = torch.clamp(sq_dists, min=0.0)

                weights = torch.exp(-sq_dists / (2.0 * (self.bandwidth ** 2))) # [S, A]
                total_weights = weights * active_weights.unsqueeze(0) # [S, A]

                denominator = torch.sum(total_weights, dim=-1, keepdim=True) + 1e-8
                new_seeds = torch.matmul(total_weights, active_coords) / denominator
                seeds = new_seeds

            # 4. Final seed confidence lookup
            grid_indices = torch.round(seeds / spacing).long()
            grid_indices = torch.clamp(grid_indices, 0, X - 1)
            final_probs = sample_density[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]

            # Sort final seeds by confidence
            sorted_idx = torch.argsort(final_probs, descending=True)
            seeds = seeds[sorted_idx]
            final_probs = final_probs[sorted_idx]

            # 5. Greedy spatial deduplication to resolve overlaps
            keep_mask = torch.ones(seeds.shape[0], dtype=torch.bool, device=device)
            for idx in range(seeds.shape[0]):
                if not keep_mask[idx]:
                    continue
                other_dists = torch.sum((seeds[idx+1:] - seeds[idx]) ** 2, dim=-1).sqrt()
                clash_mask = other_dists < CLASH_LIMIT
                keep_mask[idx+1:][clash_mask] = False

            seeds = seeds[keep_mask]
            final_probs = final_probs[keep_mask]

            num_to_copy = min(seeds.shape[0], M)
            if num_to_copy > 0:
                out_coords[b, :num_to_copy] = seeds[:num_to_copy]
                out_values[b, :num_to_copy] = final_probs[:num_to_copy]
                out_mask[b, :num_to_copy] = True

        return out_coords, out_values, out_mask


# ==============================================================================
# SECTION 4: GENERALIZED REAL-DATA TRAINING & COORDINATE DECODING
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    # Select execution device (Apple Silicon MPS, NVIDIA CUDA, or CPU) early
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    print("===========================================================================")
    print(" PREPARING PDB MOLECULAR STRUCTURAL REPOSITORY ")
    print("===========================================================================")

    train_files = [download_pdb_cif(pid) for pid in TRAIN_PDB_IDS]
    test_file = download_pdb_cif(TEST_PDB_ID)

    print(f"\nSuccessfully cached {len(train_files)} training structures and 1 unseen test structure.")

    # Load PDB coordinates once into memory to make dynamic cropping fast
    print("\nLoading PDB structures into memory once...")
    train_structures = []
    for filepath in train_files:
        structure = gemmi.read_structure(filepath)
        all_atoms = []
        carbon_atoms = []
        for struct_model in structure:
            for chain in struct_model:
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

    # Dynamic cropping function
    def crop_and_rasterize_dynamic(structures: list, box_size: float = DEFAULT_BOX_SIZE, grid_size: int = DEFAULT_GRID_SIZE, noise_level: float = TRAIN_NOISE_LEVEL, return_coords: bool = False) -> tuple:
        # Pick a random structure
        all_coords, carbon_coords = random.choice(structures)

        num_atoms = all_coords.shape[0]
        random_idx = torch.randint(0, num_atoms, (1,)).item()
        center_atom = all_coords[random_idx]

        half_box = box_size / 2.0
        carbon_mask = torch.all((carbon_coords >= center_atom - half_box) & (carbon_coords <= center_atom + half_box), dim=-1)
        cropped_carbon = carbon_coords[carbon_mask]

        all_mask = torch.all((all_coords >= center_atom - half_box) & (all_coords <= center_atom + half_box), dim=-1)
        cropped_all = all_coords[all_mask]

        # Shift coordinates to align inside the [0, box_size]^3 space
        cropped_all_centered = cropped_all - center_atom + half_box
        cropped_carbon_centered = cropped_carbon - center_atom + half_box

        # Rasterize inputs and targets on the fly
        input_density = coords_to_density(cropped_all_centered, box_size=box_size, grid_size=grid_size, sigma=INPUT_SIGMA)
        noise = torch.randn_like(input_density) * noise_level
        input_density = F.relu(input_density + noise)

        target_density = coords_to_binary_grid(cropped_carbon_centered, box_size=box_size, grid_size=grid_size, radius=DEFAULT_RADIUS)

        if return_coords:
            return input_density, target_density, cropped_carbon_centered
        return input_density, target_density

    # Pre-cache training and validation datasets
    print(f"\nPre-caching {NUM_TRAIN_CROPS} training crops from PDB structures...")
    train_dataset = []
    for idx in range(NUM_TRAIN_CROPS):
        train_input, train_target = crop_and_rasterize_dynamic(train_structures, box_size=DEFAULT_BOX_SIZE, grid_size=TRAIN_GRID_SIZE, noise_level=TRAIN_NOISE_LEVEL)
        train_dataset.append((train_input, train_target))
        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    print(f"Pre-caching {NUM_VAL_CROPS} validation crops...")
    val_dataset = []
    for idx in range(NUM_VAL_CROPS):
        val_input, val_target = crop_and_rasterize_dynamic(train_structures, box_size=DEFAULT_BOX_SIZE, grid_size=TRAIN_GRID_SIZE, noise_level=TRAIN_NOISE_LEVEL)
        val_dataset.append((val_input, val_target))
        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    # Initialize U-Net, optimizer, and scheduler
    model = UNet3D(in_channels=1, out_channels=1, init_features=UNET_INIT_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX, eta_min=SCHEDULER_ETA_MIN)
    criterion = BCEDiceLoss()

    model.to(device)
    print(f"Running on computational device: {device}")

    peak_finder = BatchedMeanShiftPeakFinder3D(
        threshold=DEFAULT_PEAK_THRESHOLD,
        bandwidth=DEFAULT_PEAK_BANDWIDTH,
        max_peaks=DEFAULT_MAX_PEAKS,
        box_size=DEFAULT_BOX_SIZE,
        iterations=DEFAULT_PEAK_ITERATIONS
    )
    peak_finder.to(device)

    print("\n" + "="*75)
    print(" RUNNING REAL PDB U-NET GENERALIZATION TRAINING ")
    print("="*75)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        # Shuffle indices of the pre-cached training dataset
        shuffled_indices = torch.randperm(len(train_dataset))

        for i in range(0, len(train_dataset), BATCH_SIZE):
            batch_indices = shuffled_indices[i : i + BATCH_SIZE]
            batch_samples = [train_dataset[idx] for idx in batch_indices]

            inputs = torch.stack([sample[0] for sample in batch_samples]).unsqueeze(1).to(device)  # [B, 1, 32, 32, 32]
            targets = torch.stack([sample[1] for sample in batch_samples]).unsqueeze(1).to(device) # [B, 1, 32, 32, 32]

            # Apply boundary-preserving random 3D flips and rotations
            inputs, targets = augment_batch_3d(inputs, targets)

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

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | LR: {current_lr:.6f} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Evaluation on a completely unseen real protein (Crambin 1CRN)
    print("\n" + "="*75)
    print(f" EVALUATION ON UNSEEN PDB STRUCTURE: CRAMBIN ({TEST_PDB_ID.upper()}) ")
    print("="*75)

    # Load Crambin structure coordinates once to accelerate test evaluation
    test_structure = gemmi.read_structure(test_file)
    test_atoms = []
    test_carbons = []
    for struct_model in test_structure:
        for chain in struct_model:
            for residue in chain:
                res_name = residue.name.strip().upper()
                if res_name not in PROTEIN_RESIDUES:
                    continue
                for atom in residue:
                    pos = [atom.pos.x, atom.pos.y, atom.pos.z]
                    test_atoms.append(pos)
                    if atom.element.name.strip().upper() == "C":
                        test_carbons.append(pos)
    test_all_coords = torch.tensor(test_atoms, dtype=torch.float32).to(device)
    test_carbon_coords = torch.tensor(test_carbons, dtype=torch.float32).to(device)
    test_structures = [(test_all_coords, test_carbon_coords)]

    # Pre-cache test crops from Crambin to construct a robust benchmark
    print(f"Pre-caching {NUM_TEST_CROPS} unseen Crambin test crops...")
    test_dataset = []
    for _ in range(NUM_TEST_CROPS):
        test_in, test_tgt, test_coords = crop_and_rasterize_dynamic(
            test_structures, box_size=DEFAULT_BOX_SIZE, grid_size=TRAIN_GRID_SIZE, noise_level=TRAIN_NOISE_LEVEL, return_coords=True
        )
        test_dataset.append((test_in, test_tgt, test_coords))

    model.eval()
    total_gt_carbons = 0
    total_matched_carbons = 0
    total_resolved_peaks = 0

    print(f"\nEvaluating model over {len(test_dataset)} test crops...")
    with torch.no_grad():
        for test_idx, (test_input, test_target, test_gt_coords) in enumerate(test_dataset):
            test_in_batch = test_input.unsqueeze(0).unsqueeze(0).to(device)
            pred_density = F.relu(model(test_in_batch))
            pred_coords, pred_vals, pred_mask = peak_finder(pred_density)

            pred_coords = pred_coords[0].cpu()
            pred_mask = pred_mask[0].cpu()

            num_pred_peaks = pred_mask.sum().item()
            num_gt_peaks = len(test_gt_coords)

            total_resolved_peaks += num_pred_peaks
            total_gt_carbons += num_gt_peaks

            matched_count = 0
            for gt_c in test_gt_coords:
                if num_pred_peaks > 0:
                    distances = torch.norm(pred_coords[:num_pred_peaks] - gt_c.cpu(), dim=-1)
                    min_dist, min_idx = torch.min(distances, dim=0)
                    if min_dist.item() <= MATCHING_RADIUS:
                        matched_count += 1
            total_matched_carbons += matched_count

    avg_accuracy = (total_matched_carbons / total_gt_carbons) * 100 if total_gt_carbons > 0 else 0.0
    print(f"\nEvaluated over {len(test_dataset)} unseen Crambin crops.")
    print(f"Average U-Net + Peak Finder resolved peaks per crop: {total_resolved_peaks / len(test_dataset):.1f}")
    print(f"Total Ground Truth Carbons across all crops: {total_gt_carbons}")
    print(f"Total Matched Carbons: {total_matched_carbons}")
    print(f"\nOverall Coordinate Recovery Accuracy: {avg_accuracy:.1f}% ({total_matched_carbons}/{total_gt_carbons} carbons resolved within {MATCHING_RADIUS} Å)")
    print("="*75)
