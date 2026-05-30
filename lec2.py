import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb

from tqdm import tqdm


# ==============================================================================
# CONSTANTS & CONFIGURATIONS
# ==============================================================================

# Seed for reproducibility
RANDOM_SEED = 42

PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}
NUCLEIC_RESIDUES = {"A", "C", "G", "U", "DA", "DC", "DG", "DT"} #D stands for deoxy
ALL_RESIDUES = PROTEIN_RESIDUES | NUCLEIC_RESIDUES
RESIDUE_MAP = {res: idx + 1 for idx, res in enumerate(sorted(list(ALL_RESIDUES)))}  # Class 0 is reserved for background

# Directories and Dataset Paths
DEFAULT_SAVE_DIR = "./pdb_data"
TRAIN_PDB_IDS = []  # Deprecated in favor of ALL_PDB_IDS K-Fold cross validation
TEST_PDB_IDS = []   # Deprecated in favor of ALL_PDB_IDS K-Fold cross validation

ALL_PDB_IDS = [
    "1ubq", "1a8o", "1bpi", "1cjg", "1eyy", "1hel", "1l2y", "1pga",
    "1shg", "1csp", "1a70", "1f9g", "2igd", "1ten", "1ycr", "3gbw",
    "1uzx", "2h3l", "1a62", "1mbo", "1bna", "1ehz", "1crn", "2cro",
    "1r69", "1acx"
]

# K-Fold Cross-Validation Configuration
K_FOLDS = 10
CURRENT_FOLD = 0  # Set to an integer 0-9 to train/evaluate a single fold, or None to loop over all folds

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
NUM_TEST_CROPS = 30     # Number of pre-cached unseen test crops per protein

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

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a CUDA-enabled GPU.")

torch.set_default_device("cuda")
print("Using device: cuda")

# ==============================================================================
# UTILITIES, PIPELINES & DATA AUGMENTATION
# ==============================================================================

def download_pdb_cif(pdb_id: str) -> str:
    path = rcsb.fetch(pdb_id, "cif", DEFAULT_SAVE_DIR)
    if isinstance(path, list):
        return str(path[0])
    return str(path)


def load_coords_biotite(filepath: str) -> tuple[torch.Tensor, torch.Tensor]:
    atoms: Any = pdbx.get_structure(pdbx.CIFFile.read(filepath), model=1)
    valid_atoms: Any = atoms[np.isin(atoms.res_name, list(ALL_RESIDUES))]
    all_coords = torch.tensor(valid_atoms.coord, dtype=torch.float32)
    res_indices = torch.tensor([RESIDUE_MAP[name] for name in valid_atoms.res_name], dtype=torch.long)
    return all_coords, res_indices


def coords_to_density(coords: torch.Tensor, sigma: float) -> torch.Tensor:
    # Generate the 3D grid ticks
    ticks = torch.linspace(0.0, DEFAULT_BOX_SIZE, GRID_SIZE)
    grid_x, grid_y, grid_z = torch.meshgrid(ticks, ticks, ticks, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1) # Shape: [G, G, G, 3]
    g_flat = grid.view(-1, 3) # Shape: [G^3, 3]

    g2 = torch.sum(g_flat ** 2, dim=-1, keepdim=True) # Shape: [G^3, 1]
    density_flat = torch.zeros(g_flat.shape[0])

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
    # Generate the 3D grid ticks
    ticks = torch.linspace(0.0, DEFAULT_BOX_SIZE, GRID_SIZE)
    grid_x, grid_y, grid_z = torch.meshgrid(ticks, ticks, ticks, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1) # Shape: [G, G, G, 3]
    g_flat = grid.view(-1, 3) # Shape: [G^3, 3]

    g2 = torch.sum(g_flat ** 2, dim=-1, keepdim=True) # Shape: [G^3, 1]
    min_dists_flat = torch.full((g_flat.shape[0],), float('inf'))

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
    # Generate the 3D grid ticks
    ticks = torch.linspace(0.0, DEFAULT_BOX_SIZE, GRID_SIZE)
    grid_x, grid_y, grid_z = torch.meshgrid(ticks, ticks, ticks, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1) # Shape: [G, G, G, 3]
    g_flat = grid.view(-1, 3) # Shape: [G^3, 3]

    g2 = torch.sum(g_flat ** 2, dim=-1, keepdim=True) # Shape: [G^3, 1]
    min_dists_flat = torch.full((g_flat.shape[0],), float('inf'))
    nearest_indices = torch.full((g_flat.shape[0],), -1, dtype=torch.long)

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
    residue_grid_flat = torch.zeros(g_flat.shape[0], dtype=torch.long)
    valid_mask = min_dists_flat <= DEFAULT_RADIUS
    residue_grid_flat[valid_mask] = res_indices[nearest_indices[valid_mask]]

    return residue_grid_flat.view(GRID_SIZE, GRID_SIZE, GRID_SIZE)


def augment_batch_3d_joint(
    inputs: torch.Tensor,
    atom_targets: torch.Tensor,
    residue_targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def get_k_folds(items: list, k: int = 10, seed: int = 42) -> list[list]:
    # Shuffle items deterministically
    random_gen = random.Random(seed)
    shuffled_items = list(items)
    random_gen.shuffle(shuffled_items)

    folds = [[] for _ in range(k)]
    for idx, item in enumerate(shuffled_items):
        folds[idx % k].append(item)
    return folds


def get_fold_split(folds: list[list], fold_idx: int) -> tuple[list, list, list]:
    k = len(folds)
    test_idx = fold_idx
    val_indices = [(fold_idx + 1) % k, (fold_idx + 2) % k]

    test_set = folds[test_idx]
    val_set = []
    for vi in val_indices:
        val_set.extend(folds[vi])

    train_set = []
    for idx in range(k):
        if idx != test_idx and idx not in val_indices:
            train_set.extend(folds[idx])

    return train_set, val_set, test_set


# ==============================================================================
# SECTION 2: THE 3D U-NET ARCHITECTURE (Volumetric Segmenter)
# ==============================================================================

def get_emb(x: torch.Tensor) -> torch.Tensor:
    emb = torch.stack((x.sin(), x.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding3D(nn.Module):
    inv_freq: torch.Tensor
    cached_penc: torch.Tensor | None

    def __init__(self, channels):
        super().__init__()
        self.channels = (channels + 5) // 6 * 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            dtype=tensor.dtype,
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc


class GeGLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Combined linear projections $xW + b$ and $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * self.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),
            nn.Linear(d_model * d_mult, d_model),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_head: int,
    ):
        super().__init__()
        self.n_heads = num_heads
        self.d_head = dim_head
        d_attn = dim_head * num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_attn, bias=False)
        self.o_proj = nn.Linear(d_attn, d_model)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.size()

        # Get query, key and value vectors
        qkv = self.qkv_proj(x)  # Shape: (batch_size, seq_len, 3*d_attn)
        q, k, v = qkv.chunk(3, dim=-1)  # Shape: (batch_size, seq_len, d_attn)

        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        output = F.scaled_dot_product_attention(q, k, v)
        output = output.permute(0, 2, 1, 3)

        return self.o_proj(output.reshape(batch_size, seq_len, -1))


class BasicTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.attn1 = SelfAttention(d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = self.attn1(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x


class SpatialTransformerBlock3d(nn.Module):
    def __init__(self, channels: int, n_heads: int, n_layers: int):
        super().__init__()
        num_groups = min(32, channels)
        if channels % num_groups != 0:
            num_groups = 1
        self.norm = torch.nn.GroupNorm(
            num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.positional_encoding = PositionalEncoding3D(channels)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(channels, n_heads, channels // n_heads)
                for _ in range(n_layers)
            ]
        )
        self.proj_out = nn.Conv3d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor):
        b, c, h, w, d = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 4, 1)

        pos_emb = self.positional_encoding(x)
        x += pos_emb

        x = x.view(b, h * w * d, c)

        for block in self.transformer_blocks:
            x = block(x)

        x = x.view(b, h, w, d, c).permute(0, 4, 1, 2, 3)
        x = self.proj_out(x)
        return x + x_in


class ConvBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels=None, kernel_size=3):
        super().__init__()
        if out_channels is None:
            out_channels = channels

        # Determine num_groups safely.
        # For the input layer (channels=1), we use 1 group (InstanceNorm).
        # For model hidden states (multiples of 32), we use standard 32 groups.
        groups_in = 32 if channels % 32 == 0 else 1
        groups_out = 32 if out_channels % 32 == 0 else 1

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups_in, num_channels=channels),
            nn.SiLU(),
            nn.Conv3d(channels, out_channels, 3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups_out, num_channels=out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                groups=out_channels,
            ),
        )

        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv3d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class DownSample3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UpSample3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.scale_factor = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_features: int = 32) -> None:
        super().__init__()
        F_dim = init_features

        # --- Encoder (Downsampling Path) ---
        self.down1 = ConvBlock3d(in_channels, F_dim)
        self.pool1 = DownSample3d(F_dim)

        self.down2 = ConvBlock3d(F_dim, F_dim * 2)
        self.pool2 = DownSample3d(F_dim * 2)

        # --- Bottleneck ---
        self.bottleneck = ConvBlock3d(F_dim * 2, F_dim * 4)
        self.bottleneck_att = SpatialTransformerBlock3d(F_dim * 4, n_heads=2, n_layers=1)

        # --- Decoder (Upsampling Path) ---
        self.up1 = UpSample3d(F_dim * 4)
        self.conv_up1 = ConvBlock3d(F_dim * 6, F_dim * 2) # 4 (upsampled) + 2 (skip connection)
        self.att1 = SpatialTransformerBlock3d(F_dim * 2, n_heads=2, n_layers=1)

        self.up2 = UpSample3d(F_dim * 2)
        self.conv_up2 = ConvBlock3d(F_dim * 3, F_dim) # 2 (upsampled) + 1 (skip connection)
        self.att2 = SpatialTransformerBlock3d(F_dim, n_heads=2, n_layers=1)

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
    def __init__(self) -> None:
        super().__init__()

    def forward(self, density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, X, Y, Z = density.shape

        M = DEFAULT_MAX_PEAKS
        out_coords = torch.zeros((B, M, 3), dtype=torch.float32)
        out_values = torch.zeros((B, M), dtype=torch.float32)
        out_mask = torch.zeros((B, M), dtype=torch.bool)

        ticks = torch.linspace(0.0, DEFAULT_BOX_SIZE, X)
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
            keep_mask = torch.ones(seeds.shape[0], dtype=torch.bool)
            for idx in range(seeds.shape[0]):
                if not keep_mask[idx]:
                    continue
                other_dists = torch.sum((seeds[idx+1:] - seeds[idx]) ** 2, dim=-1).sqrt()
                clash_mask = other_dists < CLASH_LIMIT
                keep_mask[idx+1:][clash_mask] = False

            seeds = seeds[keep_mask]
            final_probs = final_probs[keep_mask]

            num_to_copy = min(seeds.shape[0], M)
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

    print("===========================================================================")
    print(" PREPARING PDB MOLECULAR STRUCTURAL REPOSITORY ")
    print("===========================================================================")

    # Download all PDB structures upfront to guarantee local caching
    pdb_files = {pid: download_pdb_cif(pid) for pid in ALL_PDB_IDS}
    print(f"\nSuccessfully cached {len(pdb_files)} PDB structures in local repository.")

    # Load all PDB coordinates once into memory to make dynamic splits and cropping fast
    print("\nLoading PDB structures into memory once...")
    all_structures = {}
    for pid, filepath in pdb_files.items():
        all_coords, res_indices = load_coords_biotite(filepath)
        all_structures[pid] = (all_coords, res_indices)
        print(f"  Loaded {pid.upper()} | Atoms: {len(all_coords)}")

    # Partition the unified dataset into 10 folds
    folds = get_k_folds(ALL_PDB_IDS, K_FOLDS, RANDOM_SEED)

    if CURRENT_FOLD is not None:
        folds_to_run = [CURRENT_FOLD]
    else:
        folds_to_run = list(range(K_FOLDS))

    # To accumulate global test results across all folds
    all_folds_pdb_results = []
    global_gt_atoms = 0
    global_matched_atoms = 0
    global_correct_residues = 0
    global_resolved_peaks = 0
    global_num_crops = 0

    for fold_idx in folds_to_run:
        print("\n" + "="*75)
        print(f" RUNNING FOLD {fold_idx + 1}/{K_FOLDS} (70/20/10 SPLIT) ")
        print("="*75)

        # Get splits for this fold
        train_pids, val_pids, test_pids = get_fold_split(folds, fold_idx)
        print(f"Train structures ({len(train_pids)}): {[p.upper() for p in train_pids]}")
        print(f"Validation structures ({len(val_pids)}): {[p.upper() for p in val_pids]}")
        print(f"Test structures ({len(test_pids)}): {[p.upper() for p in test_pids]}")

        # Construct subsets with zero leakage
        train_structures = [all_structures[pid] for pid in train_pids]
        val_structures = [all_structures[pid] for pid in val_pids]
        test_structures = [all_structures[pid] for pid in test_pids]

        # Pre-cache validation dataset from val_structures ONLY (no leakage!)
        print(f"\nPre-caching {NUM_VAL_CROPS} validation crops from validation structures...")
        val_dataset = []
        for idx in tqdm(range(NUM_VAL_CROPS), desc="Pre-caching Val Crops"):
            val_input, val_target_atom, val_target_res = crop_and_rasterize_dynamic(val_structures, is_training=True)
            val_dataset.append((val_input, val_target_atom, val_target_res))
            if (idx + 1) % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Initialize Models, Optimizer, Scheduler, and Losses for this fold
        atom_model = UNet3D(in_channels=1, out_channels=1, init_features=UNET_INIT_FEATURES)
        residue_model = UNet3D(in_channels=1, out_channels=len(RESIDUE_MAP) + 1, init_features=UNET_INIT_FEATURES)

        optimizer = torch.optim.Adam(
            itertools.chain(atom_model.parameters(), residue_model.parameters()),
            lr=LEARNING_RATE
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX, eta_min=SCHEDULER_ETA_MIN)

        criterion_atom = BCEDiceLoss()
        criterion_residue = nn.CrossEntropyLoss()
        peak_finder = BatchedMeanShiftPeakFinder3D()

        # Train loop
        best_val_loss = float('inf')
        epochs_no_improve = 0

        steps_per_epoch = NUM_TRAIN_CROPS // BATCH_SIZE

        print(f"\nTraining models on fold dataset (Epochs: {NUM_EPOCHS})...")
        for epoch in range(1, NUM_EPOCHS + 1):
            atom_model.train()
            residue_model.train()
            train_loss = 0.0

            step_bar = tqdm(
                range(steps_per_epoch),
                desc=f"Epoch {epoch:02d}/{NUM_EPOCHS}",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
                leave=True
            )
            for step in step_bar:
                # Sample crops on-the-fly from train_structures ONLY (no leakage!)
                batch_samples = [crop_and_rasterize_dynamic(train_structures, is_training=True) for _ in range(BATCH_SIZE)]

                inputs = torch.stack([sample[0] for sample in batch_samples]).unsqueeze(1)
                atom_targets = torch.stack([sample[1] for sample in batch_samples]).unsqueeze(1)
                residue_targets = torch.stack([sample[2] for sample in batch_samples]).long()

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

                step_loss = loss.item()
                train_loss += step_loss * BATCH_SIZE
                step_bar.set_postfix(loss=f"{step_loss:.4f}")

            train_loss /= (steps_per_epoch * BATCH_SIZE)

            # Validation evaluation on the stable, pre-cached set
            atom_model.eval()
            residue_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for idx, (val_input, val_target_atom, val_target_res) in enumerate(val_dataset):
                    val_input_tensor = val_input.unsqueeze(0).unsqueeze(0)
                    val_target_atom_tensor = val_target_atom.unsqueeze(0).unsqueeze(0)
                    val_target_res_tensor = val_target_res.unsqueeze(0)

                    # Forward pass with deep supervision enabled
                    val_atom_pred, val_atom_ds = atom_model(val_input_tensor, return_ds=True)
                    val_res_pred, val_res_ds = residue_model(val_input_tensor, return_ds=True)

                    val_atom_pred = torch.sigmoid(val_atom_pred)

                    # Main losses at 32^3 scale
                    loss_atom_main = criterion_atom(val_atom_pred, val_target_atom_tensor)
                    loss_residue_main = criterion_residue(val_res_pred, val_target_res_tensor)

                    # Deep supervision losses at 16^3 scale
                    val_atom_targets_ds = F.max_pool3d(val_target_atom_tensor, kernel_size=2, stride=2)
                    val_residue_targets_ds = F.max_pool3d(val_target_res_tensor.float().unsqueeze(1), kernel_size=2, stride=2).squeeze(1).long()

                    loss_atom_ds = criterion_atom(torch.sigmoid(val_atom_ds), val_atom_targets_ds)
                    loss_residue_ds = criterion_residue(val_res_ds, val_residue_targets_ds)

                    # Joint validation loss matching the training weights
                    loss_atom = loss_atom_main + 0.5 * loss_atom_ds
                    loss_residue = loss_residue_main + 0.5 * loss_residue_ds

                    val_loss += (loss_atom + loss_residue).item()
                val_loss /= len(val_dataset)

            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            tqdm.write(f"Epoch {epoch:02d}/{NUM_EPOCHS} | LR: {current_lr:.6f} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

            # Check for early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered: validation loss did not improve for {EARLY_STOPPING_PATIENCE} consecutive epochs.")
                    break

        # Test Evaluation for this fold
        print("\n" + "-"*75)
        print(f" EVALUATING FOLD {fold_idx + 1} ON UNSEEN TEST STRUCTURES ")
        print("-"*75)

        atom_model.eval()
        residue_model.eval()

        fold_pdb_results = []

        for pid in test_pids:
            print(f"Evaluating target PDB: {pid.upper()}")
            # Get pre-loaded structures for this test target
            test_target_structure = [all_structures[pid]]

            print(f"  Evaluating over {NUM_TEST_CROPS} test crops on-the-fly...")

            pid_gt_atoms = 0
            pid_matched_atoms = 0
            pid_correct_residues = 0
            pid_resolved_peaks = 0

            with torch.no_grad():
                for test_idx in tqdm(range(NUM_TEST_CROPS), desc=f"Evaluating {pid.upper()}", leave=False):
                    test_input, test_target_atom, test_target_res, test_gt_coords, test_gt_res_indices = crop_and_rasterize_dynamic(
                        test_target_structure, return_coords=True, is_training=False
                    )
                    test_in_batch = test_input.unsqueeze(0).unsqueeze(0)

                    # Predict atom density and resolve peaks
                    pred_density = F.relu(torch.sigmoid(atom_model(test_in_batch)))
                    pred_coords, pred_vals, pred_mask = peak_finder(pred_density)

                    # Predict residue classes
                    pred_res_logits = residue_model(test_in_batch)

                    pred_coords = pred_coords[0].cpu()
                    pred_mask = pred_mask[0].cpu()

                    num_pred_peaks = pred_mask.sum().item()
                    num_gt_peaks = len(test_gt_coords)

                    pid_resolved_peaks += num_pred_peaks
                    pid_gt_atoms += num_gt_peaks

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
                                pred_class = torch.argmax(logits[1:]).item() + 1
                                if pred_class == gt_res_idx:
                                    correct_res_count += 1

                    pid_matched_atoms += matched_count
                    pid_correct_residues += correct_res_count

            avg_peaks_per_crop = pid_resolved_peaks / NUM_TEST_CROPS if NUM_TEST_CROPS > 0 else 0.0
            recovery_pct = (pid_matched_atoms / pid_gt_atoms) * 100 if pid_gt_atoms > 0 else 0.0
            class_pct = (pid_correct_residues / pid_matched_atoms) * 100 if pid_matched_atoms > 0 else 0.0

            print(f"  Target {pid.upper()} | Avg Peaks/Crop: {avg_peaks_per_crop:.1f} | Recovery: {recovery_pct:.1f}% | Classification: {class_pct:.1f}%")

            result_entry = {
                "fold": fold_idx + 1,
                "pid": pid.upper(),
                "avg_peaks": avg_peaks_per_crop,
                "gt_atoms": pid_gt_atoms,
                "matched_atoms": pid_matched_atoms,
                "recovery_pct": recovery_pct,
                "class_pct": class_pct
            }
            fold_pdb_results.append(result_entry)
            all_folds_pdb_results.append(result_entry)

            global_gt_atoms += pid_gt_atoms
            global_matched_atoms += pid_matched_atoms
            global_correct_residues += pid_correct_residues
            global_resolved_peaks += pid_resolved_peaks
            global_num_crops += NUM_TEST_CROPS

        # Print Fold Summary
        print(f"\n--- FOLD {fold_idx + 1} SUMMARY ---")
        print(" PDB ID  | Resolved/Crop | Total Atoms | Recovery % | Classification %")
        print("-" * 65)
        for res in fold_pdb_results:
            print(f" {res['pid']:<7} | {res['avg_peaks']:<13.1f} | {res['gt_atoms']:<11} | {res['recovery_pct']:<10.1f}% | {res['class_pct']:<16.1f}%")
        print("-" * 65)

    # Final Cross-Validation Summary Table
    print("\n" + "="*85)
    print(" FINAL K-FOLD CROSS-VALIDATION (70/20/10 SPLIT) SUMMARY ")
    print("="*85)
    print(" Fold | PDB ID  | Resolved/Crop | Total Atoms | Recovery % | Classification %")
    print("-" * 85)
    for res in all_folds_pdb_results:
        print(f" {res['fold']:<4} | {res['pid']:<7} | {res['avg_peaks']:<13.1f} | {res['gt_atoms']:<11} | {res['recovery_pct']:<10.1f}% | {res['class_pct']:<16.1f}%")
    print("-" * 85)

    global_avg_peaks = global_resolved_peaks / global_num_crops if global_num_crops > 0 else 0.0
    global_recovery = (global_matched_atoms / global_gt_atoms) * 100 if global_gt_atoms > 0 else 0.0
    global_classification = (global_correct_residues / global_matched_atoms) * 100 if global_matched_atoms > 0 else 0.0

    print(f" {'OVERALL':<12} | {global_avg_peaks:<13.1f} | {global_gt_atoms:<11} | {global_recovery:<10.1f}% | {global_classification:<16.1f}%")
    print("="*85)
