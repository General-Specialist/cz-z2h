# DO NOT EXECUTE THIS SCRIPT LOCALLY FOR ANY REASON WHATSOEVER

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb

# ==============================================================================
# CONSTANTS & CONFIGURATIONS
# ==============================================================================

# Seed for reproducibility
RANDOM_SEED = 42

# Standard amino acid residues
PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}

# Standard nucleic acid residues (both RNA and DNA)
NUCLEIC_RESIDUES = {"A", "C", "G", "U", "DA", "DC", "DG", "DT"}

# Merge all valid residues (protein + nucleic acids) to filter out water, ions, and ligands
ALL_RESIDUES = PROTEIN_RESIDUES | NUCLEIC_RESIDUES

# Stable, sorted list of standard residues for multi-class indexing
RESIDUE_LIST = sorted(list(ALL_RESIDUES))
RESIDUE_MAP = {res: idx + 1 for idx, res in enumerate(RESIDUE_LIST)}  # Class 0 is reserved for background

# Directories and Dataset Paths
DEFAULT_SAVE_DIR = "./pdb_data"
TRAIN_PDB_IDS = [
    "1ubq", "1a8o", "1bpi", "1cjg", "1eyy", "1hel", "1l2y", "1pga",
    "1shg", "1csp", "1a70", "1f9g", "2igd", "1ten", "1ycr", "3gbw",
    "1uzx", "2h3l", "1a62", "1mbo", "1bna", "1ehz"
]
TEST_PDB_ID = "1crn"

# Volumetric & Rasterization Grid Configuration
DEFAULT_BOX_SIZE = 8.0
GRID_SIZE = 32

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
UNET_IN_CHANNELS = 1
UNET_OUT_CHANNELS = 1
UNET_INIT_FEATURES = 32
LEARNING_RATE = 0.001
SCHEDULER_T_MAX = 60
SCHEDULER_ETA_MIN = 1e-5
NUM_EPOCHS = 60
BATCH_SIZE = 8
EARLY_STOPPING_PATIENCE = 5

# Dataset Configuration
NUM_TRAIN_CROPS = 2000  # Number of virtual crops sampled on-the-fly per epoch
NUM_VAL_CROPS = 400     # Number of pre-cached validation crops
NUM_TEST_CROPS = 50     # Number of pre-cached unseen test crops

# 3D Peak Finding & Post-Processing (Mean-Shift)
DEFAULT_PEAK_THRESHOLD = 0.30
DEFAULT_PEAK_BANDWIDTH = 1.0
DEFAULT_MAX_PEAKS = 128
DEFAULT_PEAK_ITERATIONS = 5
CLASH_LIMIT = 0.6

# Coordinate Metric Evaluation
MATCHING_RADIUS = 1.0

# Data Augmentation Constants
AUGMENT_FLIP_PROB = 0.5

# Loss Configuration
BCE_DICE_EPS = 1e-6

# Network Architectural Constants
CHANNEL_ATTN_REDUCTION = 4
SPATIAL_ATTN_KERNEL_SIZE = 3
SPATIAL_ATTN_PADDING = 1
CONV_KERNEL_SIZE = 3
CONV_PADDING = 1
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2
TRANSPOSE_KERNEL_SIZE = 2
TRANSPOSE_STRIDE = 2
OUT_CONV_KERNEL_SIZE = 1

# Peak Finder Numerical Stability & Filter Constants
MAXPOOL_PEAK_KERNEL_SIZE = 3
MAXPOOL_PEAK_STRIDE = 1
MAXPOOL_PEAK_PADDING = 1
PEAK_FINDER_EPSILON = 1e-8


# ==============================================================================
# UTILITIES, PIPELINES & DATA AUGMENTATION
# ==============================================================================

def download_pdb_cif(pdb_id: str) -> str:
    return rcsb.fetch(pdb_id, "cif", DEFAULT_SAVE_DIR)


def load_coords_biotite(filepath: str) -> tuple[torch.Tensor, torch.Tensor]:
    atoms = pdbx.get_structure(pdbx.CIFFile.read(filepath), model=1)
    valid_atoms = atoms[np.isin(atoms.res_name, list(ALL_RESIDUES))]
    all_coords = torch.tensor(valid_atoms.coord, dtype=torch.float32)
    res_indices = torch.tensor([RESIDUE_MAP[name] for name in valid_atoms.res_name], dtype=torch.long)
    return all_coords, res_indices


def load_and_crop_pdb(filepath: str) -> tuple[torch.Tensor, torch.Tensor]:
    all_coords, res_indices = load_coords_biotite(filepath)

    # Select a random atom as the local crop anchor
    num_atoms = all_coords.shape[0]
    random_idx = torch.randint(0, num_atoms, (1,)).item()
    center_atom = all_coords[random_idx]

    # Keep coordinates falling inside the local box bounds around center_atom
    half_box = DEFAULT_BOX_SIZE / 2.0
    mask = torch.all((all_coords >= center_atom - half_box) & (all_coords <= center_atom + half_box), dim=-1)

    cropped_all = all_coords[mask]
    cropped_res_indices = res_indices[mask]

    # Shift coordinates so that center_atom is centered exactly at [DEFAULT_BOX_SIZE/2, DEFAULT_BOX_SIZE/2, DEFAULT_BOX_SIZE/2]
    # This maps the cropped region exactly into the [0, DEFAULT_BOX_SIZE]^3 voxel space.
    cropped_all_centered = cropped_all - center_atom + half_box

    return cropped_all_centered, cropped_res_indices


def coords_to_density(coords: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Vectorized, differentiable, and chunked 3D density rasterization.
    Processes coordinates in chunks to limit GPU memory footprint and prevent CUDA OOM.
    """
    if coords.shape[0] == 0:
        return torch.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), device=coords.device)

    device = coords.device
    # Generate the 3D grid ticks
    ticks = torch.linspace(0.0, DEFAULT_BOX_SIZE, GRID_SIZE, device=device)
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

    return density_flat.view(GRID_SIZE, GRID_SIZE, GRID_SIZE)


def coords_to_binary_grid(coords: torch.Tensor) -> torch.Tensor:
    """
    Vectorized, differentiable, and chunked 3D binary grid rasterizer.
    Processes coordinates in chunks to limit GPU memory footprint and prevent CUDA OOM.
    """
    if coords.shape[0] == 0:
        return torch.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), device=coords.device)

    device = coords.device
    # Generate the 3D grid ticks
    ticks = torch.linspace(0.0, DEFAULT_BOX_SIZE, GRID_SIZE, device=device)
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

    binary_grid_flat = (min_dists_flat <= DEFAULT_RADIUS).float()
    return binary_grid_flat.view(GRID_SIZE, GRID_SIZE, GRID_SIZE)


def coords_to_residue_grid(coords: torch.Tensor, res_indices: torch.Tensor) -> torch.Tensor:
    """
    Vectorized, differentiable, and chunked 3D residue grid rasterizer.
    Processes coordinates in chunks to limit GPU memory footprint and prevent CUDA OOM.
    Assigns each voxel the class index of the closest atom within DEFAULT_RADIUS,
    otherwise 0 (background).
    """
    if coords.shape[0] == 0:
        return torch.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=torch.long, device=coords.device)

    device = coords.device
    # Generate the 3D grid ticks
    ticks = torch.linspace(0.0, DEFAULT_BOX_SIZE, GRID_SIZE, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(ticks, ticks, ticks, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1) # Shape: [G, G, G, 3]
    g_flat = grid.view(-1, 3) # Shape: [G^3, 3]

    g2 = torch.sum(g_flat ** 2, dim=-1, keepdim=True) # Shape: [G^3, 1]
    min_dists_flat = torch.full((g_flat.shape[0],), float('inf'), device=device)
    nearest_indices = torch.full((g_flat.shape[0],), -1, dtype=torch.long, device=device)

    # Process atoms in chunks to cap GPU memory footprint
    chunk_size = RASTER_CHUNK_SIZE
    for i in range(0, coords.shape[0], chunk_size):
        c_chunk = coords[i : i + chunk_size]
        c2_chunk = torch.sum(c_chunk ** 2, dim=-1, keepdim=True).t() # Shape: [1, chunk]

        sq_dists_chunk = g2 + c2_chunk - 2.0 * torch.matmul(g_flat, c_chunk.t())
        sq_dists_chunk = torch.clamp(sq_dists_chunk, min=0.0)
        dists_chunk = sq_dists_chunk.sqrt()

        chunk_min, chunk_arg = torch.min(dists_chunk, dim=-1)
        update_mask = chunk_min < min_dists_flat
        min_dists_flat[update_mask] = chunk_min[update_mask]
        nearest_indices[update_mask] = chunk_arg[update_mask] + i

    # Map nearest indices to residue indices
    residue_grid_flat = torch.zeros(g_flat.shape[0], dtype=torch.long, device=device)
    valid_mask = min_dists_flat <= DEFAULT_RADIUS
    residue_grid_flat[valid_mask] = res_indices[nearest_indices[valid_mask]]

    return residue_grid_flat.view(GRID_SIZE, GRID_SIZE, GRID_SIZE)


def augment_batch_3d_joint(
    inputs: torch.Tensor,
    atom_targets: torch.Tensor,
    residue_targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies boundary-preserving random 3D rotations and flips to the batch.
    Applies the exact same transformation to the input, atom detection targets, and residue classification targets.
    """
    B = inputs.shape[0]
    augmented_inputs = []
    augmented_atoms = []
    augmented_residues = []

    for b in range(B):
        x = inputs[b]
        y_atom = atom_targets[b]
        y_res = residue_targets[b]

        # 1. Random Flips (Reflections) - boundary-preserving
        for dim in (-3, -2, -1):
            if random.random() > AUGMENT_FLIP_PROB:
                x = torch.flip(x, dims=[dim])
                y_atom = torch.flip(y_atom, dims=[dim])
                y_res = torch.flip(y_res, dims=[dim])

        # 2. Random 90-degree Rotations - boundary-preserving
        for plane in [(-3, -2), (-2, -1), (-3, -1)]:
            k = random.randint(0, 3)
            if k > 0:
                x = torch.rot90(x, k, dims=plane)
                y_atom = torch.rot90(y_atom, k, dims=plane)
                y_res = torch.rot90(y_res, k, dims=plane)

        augmented_inputs.append(x)
        augmented_atoms.append(y_atom)
        augmented_residues.append(y_res)

    return torch.stack(augmented_inputs), torch.stack(augmented_atoms), torch.stack(augmented_residues)


class BCEDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction='mean')

        # Soft Dice Loss
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)

        intersection = torch.sum(pred_flat * target_flat, dim=-1)
        union = torch.sum(pred_flat, dim=-1) + torch.sum(target_flat, dim=-1)

        dice = 1.0 - (2.0 * intersection + BCE_DICE_EPS) / (union + BCE_DICE_EPS)
        return bce + dice.mean()


def generate_cryo_em_sample(filepaths: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Crops a local spatial sub-volume from a random protein and rasterizes:
    - Input: all atoms, wider blur (lower resolution), with added Gaussian noise.
    - Target 1: all atoms, sharp blur, clean.
    - Target 2: residue class voxel grid.
    """
    filepath = random.choice(filepaths)
    all_coords, res_indices = load_and_crop_pdb(filepath)

    # Input map: all atoms, wider blur (simulating low-resolution cryo-EM map)
    input_density = coords_to_density(all_coords, sigma=INPUT_SIGMA)
    noise = torch.randn_like(input_density) * DEFAULT_NOISE_LEVEL
    input_density = F.relu(input_density + noise) # clamp negative densities to 0

    # Target 1 map: all atoms, binary grid
    target_density = coords_to_binary_grid(all_coords)

    # Target 2 map: residue class grid
    target_residue = coords_to_residue_grid(all_coords, res_indices)

    return input_density, target_density, target_residue, all_coords


# ==============================================================================
# SECTION 2: THE 3D U-NET ARCHITECTURE (Volumetric Segmenter)
# ==============================================================================

class ChannelAttention3D(nn.Module):
    """
    3D Squeeze-and-Excitation Channel Attention module.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // CHANNEL_ATTN_REDUCTION),
            nn.ReLU(inplace=True),
            nn.Linear(channels // CHANNEL_ATTN_REDUCTION, channels),
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
        self.conv = nn.Conv3d(2, 1, kernel_size=SPATIAL_ATTN_KERNEL_SIZE, padding=SPATIAL_ATTN_PADDING)

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
            nn.Conv3d(in_channels, out_channels, kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    """
    Fully batched and parameterized 3D U-Net Segmenter with Channel & Spatial Attention.
    Supports Deep Supervision for training stability.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_features: int = 32) -> None:
        super().__init__()
        F_dim = init_features

        # --- Encoder (Downsampling Path) ---
        self.down1 = DoubleConv3D(in_channels, F_dim)
        self.pool1 = nn.MaxPool3d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)

        self.down2 = DoubleConv3D(F_dim, F_dim * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)

        # --- Bottleneck ---
        self.bottleneck = DoubleConv3D(F_dim * 2, F_dim * 4)
        self.bottleneck_att = ChannelAttention3D(F_dim * 4)

        # --- Decoder (Upsampling Path) ---
        self.up1 = nn.ConvTranspose3d(F_dim * 4, F_dim * 2, kernel_size=TRANSPOSE_KERNEL_SIZE, stride=TRANSPOSE_STRIDE)
        self.conv_up1 = DoubleConv3D(F_dim * 4, F_dim * 2)
        self.att1 = SpatialAttention3D()

        self.up2 = nn.ConvTranspose3d(F_dim * 2, F_dim, kernel_size=TRANSPOSE_KERNEL_SIZE, stride=TRANSPOSE_STRIDE)
        self.conv_up2 = DoubleConv3D(F_dim * 2, F_dim)
        self.att2 = SpatialAttention3D()

        # Auxiliary head for Deep Supervision at intermediate resolution (scale 16^3)
        self.ds_conv1 = nn.Conv3d(F_dim * 2, out_channels, kernel_size=OUT_CONV_KERNEL_SIZE)
        
        # Main output head (scale 32^3)
        self.out_conv = nn.Conv3d(F_dim, out_channels, kernel_size=OUT_CONV_KERNEL_SIZE)

    def forward(self, x: torch.Tensor, return_ds: bool = False) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # --- Encoder ---
        x1 = self.down1(x)
        p1 = self.pool1(x1)

        x2 = self.down2(p1)
        p2 = self.pool2(x2)

        # --- Bottleneck ---
        b = self.bottleneck(p2)
        b = self.bottleneck_att(b)

        # --- Decoder ---
        u1 = self.up1(b)
        c1 = torch.cat([u1, x2], dim=1) # Skip connection
        x3 = self.conv_up1(c1)
        x3 = self.att1(x3)

        u2 = self.up2(x3)
        c2 = torch.cat([u2, x1], dim=1) # Skip connection
        x4 = self.conv_up2(c2)
        x4 = self.att2(x4)

        out = self.out_conv(x4)
        
        if return_ds:
            ds_out = self.ds_conv1(x3)
            return out, ds_out
        return out


# ==============================================================================
# SECTION 3: 3D NON-MAXIMUM SUPPRESSION (Peak-Finders)
# ==============================================================================

class BatchedMeanShiftPeakFinder3D(nn.Module):
    """
    Continuous 3D Mean-Shift Clustering Peak Finder natively in PyTorch on GPU.
    Seek mode centers in real continuous space with sub-voxel precision.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, X, Y, Z = density.shape
        device = density.device

        M = DEFAULT_MAX_PEAKS
        out_coords = torch.zeros((B, M, 3), dtype=torch.float32, device=device)
        out_values = torch.zeros((B, M), dtype=torch.float32, device=device)
        out_mask = torch.zeros((B, M), dtype=torch.bool, device=device)

        ticks = torch.linspace(0.0, DEFAULT_BOX_SIZE, X, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(ticks, ticks, ticks, indexing='ij')
        grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1) # Shape: [X, Y, Z, 3]
        flat_grid = grid_coords.view(-1, 3) # Shape: [X*Y*Z, 3]
        spacing = DEFAULT_BOX_SIZE / (X - 1)

        # 3D MaxPool filter to select high-quality starting seeds
        max_pooled = F.max_pool3d(
            density,
            kernel_size=MAXPOOL_PEAK_KERNEL_SIZE,
            stride=MAXPOOL_PEAK_STRIDE,
            padding=MAXPOOL_PEAK_PADDING
        )
        is_peak_mask = (density == max_pooled) & (density > DEFAULT_PEAK_THRESHOLD)

        for b in range(B):
            sample_density = density[b, 0] # [X, Y, Z]
            sample_peaks = is_peak_mask[b, 0] # [X, Y, Z]

            # 1. Extract active coordinates and weights for distance computations
            active_mask = sample_density > DEFAULT_PEAK_THRESHOLD
            active_coords = flat_grid[active_mask.view(-1)] # [A, 3]
            active_weights = sample_density[active_mask] # [A]

            # 2. Extract local max-pooling peaks as starting seeds
            seeds = flat_grid[sample_peaks.view(-1)] # [S, 3]
            seed_probs = sample_density[sample_peaks]
            sorted_idx = torch.argsort(seed_probs, descending=True)
            seeds = seeds[sorted_idx[:M]]

            # 3. Iterative Mean-Shift seeks in continuous space
            for _ in range(DEFAULT_PEAK_ITERATIONS):
                s2 = torch.sum(seeds ** 2, dim=-1, keepdim=True) # [S, 1]
                a2 = torch.sum(active_coords ** 2, dim=-1, keepdim=True).t() # [1, A]
                sq_dists = s2 + a2 - 2.0 * torch.matmul(seeds, active_coords.t()) # [S, A]
                sq_dists = torch.clamp(sq_dists, min=0.0)

                weights = torch.exp(-sq_dists / (2.0 * (DEFAULT_PEAK_BANDWIDTH ** 2))) # [S, A]
                total_weights = weights * active_weights.unsqueeze(0) # [S, A]

                denominator = torch.sum(total_weights, dim=-1, keepdim=True) + PEAK_FINDER_EPSILON
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

# Dynamic cropping function using global constants
def crop_and_rasterize_dynamic(structures: list, return_coords: bool = False, is_training: bool = False) -> tuple:
    # Pick a random structure
    all_coords, res_indices = random.choice(structures)

    num_atoms = all_coords.shape[0]
    random_idx = torch.randint(0, num_atoms, (1,)).item()
    center_atom = all_coords[random_idx]

    half_box = DEFAULT_BOX_SIZE / 2.0
    all_mask = torch.all((all_coords >= center_atom - half_box) & (all_coords <= center_atom + half_box), dim=-1)
    cropped_all = all_coords[all_mask]
    cropped_res_indices = res_indices[all_mask]

    # Shift coordinates to align inside the [0, DEFAULT_BOX_SIZE]^3 space
    cropped_all_centered = cropped_all - center_atom + half_box

    # Sample blur sigma and noise dynamically for training robust to map qualities
    if is_training:
        sigma = random.uniform(0.8, 1.8)
        noise_level = random.uniform(0.01, 0.08)
    else:
        sigma = INPUT_SIGMA
        noise_level = TRAIN_NOISE_LEVEL

    # Input: all atoms, wider blur (simulating low-resolution cryo-EM map)
    input_density = coords_to_density(cropped_all_centered, sigma=sigma)
    noise = torch.randn_like(input_density) * noise_level
    input_density = F.relu(input_density + noise)

    # Target 1 (Atom Detection): binary grid of all atoms (clean)
    target_density = coords_to_binary_grid(cropped_all_centered)

    # Target 2 (Residue Classification): multi-class residue grid
    target_residue = coords_to_residue_grid(cropped_all_centered, cropped_res_indices)

    if return_coords:
        return input_density, target_density, target_residue, cropped_all_centered, cropped_res_indices
    return input_density, target_density, target_residue


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

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
        all_coords, res_indices = load_coords_biotite(filepath)
        train_structures.append((all_coords.to(device), res_indices.to(device)))
        print(f"  Loaded {os.path.basename(filepath)} | Atoms: {len(all_coords)}")

    # Pre-cache validation dataset (training set is sampled on-the-fly for infinite variations)
    print(f"\nPre-caching {NUM_VAL_CROPS} validation crops...")
    val_dataset = []
    for idx in range(NUM_VAL_CROPS):
        val_input, val_target_atom, val_target_res = crop_and_rasterize_dynamic(train_structures)
        val_dataset.append((val_input, val_target_atom, val_target_res))
        if (idx + 1) % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Initialize Dual U-Nets: Atom Detection & Residue Classification
    atom_model = UNet3D(in_channels=1, out_channels=1, init_features=UNET_INIT_FEATURES)
    residue_model = UNet3D(in_channels=1, out_channels=len(RESIDUE_MAP) + 1, init_features=UNET_INIT_FEATURES)

    # Move models to device
    atom_model.to(device)
    residue_model.to(device)

    # Set up joint optimizer for both models
    import itertools
    optimizer = torch.optim.Adam(
        itertools.chain(atom_model.parameters(), residue_model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX, eta_min=SCHEDULER_ETA_MIN)

    criterion_atom = BCEDiceLoss()
    criterion_residue = nn.CrossEntropyLoss()

    print(f"Running on computational device: {device}")

    peak_finder = BatchedMeanShiftPeakFinder3D()
    peak_finder.to(device)

    print("\n" + "="*75)
    print(" RUNNING REAL PDB U-NET GENERALIZATION TRAINING ")
    print("="*75)

    steps_per_epoch = NUM_TRAIN_CROPS // BATCH_SIZE

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        atom_model.train()
        residue_model.train()
        train_loss = 0.0

        for step in range(steps_per_epoch):
            # Sample crops on-the-fly from the 20 structures to get infinite diverse training inputs
            batch_samples = [crop_and_rasterize_dynamic(train_structures, is_training=True) for _ in range(BATCH_SIZE)]

            inputs = torch.stack([sample[0] for sample in batch_samples]).unsqueeze(1).to(device)
            atom_targets = torch.stack([sample[1] for sample in batch_samples]).unsqueeze(1).to(device)
            residue_targets = torch.stack([sample[2] for sample in batch_samples]).long().to(device)

            # Apply boundary-preserving random 3D flips and rotations jointly
            inputs, atom_targets, residue_targets = augment_batch_3d_joint(inputs, atom_targets, residue_targets)

            optimizer.zero_grad()
            
            # Forward pass with deep supervision enabled
            atom_preds, atom_ds = atom_model(inputs, return_ds=True)
            residue_preds, residue_ds = residue_model(inputs, return_ds=True)
            
            atom_preds = torch.sigmoid(atom_preds)

            # Main losses at 32^3 scale
            loss_atom_main = criterion_atom(atom_preds, atom_targets)
            loss_residue_main = criterion_residue(residue_preds, residue_targets)

            # Deep supervision losses at 16^3 scale
            # Downsample target grids to match 16^3 intermediate scale
            atom_targets_ds = F.max_pool3d(atom_targets, kernel_size=2, stride=2)
            residue_targets_ds = F.max_pool3d(residue_targets.float().unsqueeze(1), kernel_size=2, stride=2).squeeze(1).long()

            loss_atom_ds = criterion_atom(torch.sigmoid(atom_ds), atom_targets_ds)
            loss_residue_ds = criterion_residue(residue_ds, residue_targets_ds)

            # Total joint loss (aux losses weighted by 0.5)
            loss_atom = loss_atom_main + 0.5 * loss_atom_ds
            loss_residue = loss_residue_main + 0.5 * loss_residue_ds
            loss = loss_atom + loss_residue
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * BATCH_SIZE

        train_loss /= (steps_per_epoch * BATCH_SIZE)

        # Validation evaluation on the stable, pre-cached set
        atom_model.eval()
        residue_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_input, val_target_atom, val_target_res in val_dataset:
                val_input_tensor = val_input.unsqueeze(0).unsqueeze(0).to(device)
                val_target_atom_tensor = val_target_atom.unsqueeze(0).unsqueeze(0).to(device)
                val_target_res_tensor = val_target_res.unsqueeze(0).to(device)
                
                val_atom_pred = torch.sigmoid(atom_model(val_input_tensor))
                val_res_pred = residue_model(val_input_tensor)
                
                loss_atom = criterion_atom(val_atom_pred, val_target_atom_tensor)
                loss_residue = criterion_residue(val_res_pred, val_target_res_tensor)
                
                val_loss += (loss_atom + loss_residue).item()
            val_loss /= len(val_dataset)

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | LR: {current_lr:.6f} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Check for early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered: validation loss did not improve for {EARLY_STOPPING_PATIENCE} consecutive epochs.")
                break

    # Evaluation on a completely unseen real protein (Crambin 1CRN)
    print("\n" + "="*75)
    print(f" EVALUATION ON UNSEEN PDB STRUCTURE: CRAMBIN ({TEST_PDB_ID.upper()}) ")
    print("="*75)

    # Load Crambin structure coordinates once to accelerate test evaluation
    test_all_coords, test_res_indices = load_coords_biotite(test_file)
    test_structures = [(test_all_coords.to(device), test_res_indices.to(device))]

    # Pre-cache test crops from Crambin to construct a robust benchmark
    print(f"Pre-caching {NUM_TEST_CROPS} unseen Crambin test crops...")
    test_dataset = []
    for _ in range(NUM_TEST_CROPS):
        test_in, test_tgt_atom, test_tgt_res, test_coords, test_res_ind = crop_and_rasterize_dynamic(
            test_structures, return_coords=True
        )
        test_dataset.append((test_in, test_tgt_atom, test_tgt_res, test_coords, test_res_ind))

    atom_model.eval()
    residue_model.eval()
    total_gt_atoms = 0
    total_matched_atoms = 0
    total_correct_residues = 0
    total_resolved_peaks = 0

    print(f"\nEvaluating model over {len(test_dataset)} test crops...")
    with torch.no_grad():
        for test_idx, (test_input, test_target_atom, test_target_res, test_gt_coords, test_gt_res_indices) in enumerate(test_dataset):
            test_in_batch = test_input.unsqueeze(0).unsqueeze(0).to(device)
            
            # Predict atom density and resolve peaks
            pred_density = F.relu(torch.sigmoid(atom_model(test_in_batch)))
            pred_coords, pred_vals, pred_mask = peak_finder(pred_density)

            # Predict residue classes
            pred_res_logits = residue_model(test_in_batch) # [1, len(RESIDUE_MAP) + 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]

            pred_coords = pred_coords[0].cpu()
            pred_mask = pred_mask[0].cpu()

            num_pred_peaks = pred_mask.sum().item()
            num_gt_peaks = len(test_gt_coords)

            total_resolved_peaks += num_pred_peaks
            total_gt_atoms += num_gt_peaks

            spacing = DEFAULT_BOX_SIZE / (GRID_SIZE - 1)

            matched_count = 0
            correct_res_count = 0
            for i, gt_c in enumerate(test_gt_coords):
                gt_res_idx = test_gt_res_indices[i].item()
                if num_pred_peaks > 0:
                    distances = torch.norm(pred_coords[:num_pred_peaks] - gt_c.cpu(), dim=-1)
                    min_dist, min_idx = torch.min(distances, dim=0)
                    if min_dist.item() <= MATCHING_RADIUS:
                        matched_count += 1
                        
                        # Query the predicted residue class at this closest peak's coordinates
                        p_coord = pred_coords[min_idx.item()]
                        grid_idx = torch.round(p_coord / spacing).long()
                        grid_idx = torch.clamp(grid_idx, 0, GRID_SIZE - 1)
                        
                        logits = pred_res_logits[0, :, grid_idx[0], grid_idx[1], grid_idx[2]]
                        pred_class = torch.argmax(logits).item()
                        if pred_class == gt_res_idx:
                            correct_res_count += 1
                            
            total_matched_atoms += matched_count
            total_correct_residues += correct_res_count

    avg_recovery = (total_matched_atoms / total_gt_atoms) * 100 if total_gt_atoms > 0 else 0.0
    avg_classification = (total_correct_residues / total_matched_atoms) * 100 if total_matched_atoms > 0 else 0.0
    print(f"\nEvaluated over {len(test_dataset)} unseen Crambin crops.")
    print(f"Average U-Net + Peak Finder resolved peaks per crop: {total_resolved_peaks / len(test_dataset):.1f}")
    print(f"Total Ground Truth Atoms across all crops: {total_gt_atoms}")
    print(f"Total Matched Atoms: {total_matched_atoms}")
    print(f"Total Correct Residues (of matched): {total_correct_residues}")
    print(f"\nOverall Coordinate Recovery Accuracy: {avg_recovery:.1f}% ({total_matched_atoms}/{total_gt_atoms} atoms resolved within {MATCHING_RADIUS} Å)")
    print(f"Overall Residue Classification Accuracy: {avg_classification:.1f}% ({total_correct_residues}/{total_matched_atoms} residue types correctly matched)")
    print("="*75)
