import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb
from tqdm import tqdm

# ==============================================================================
# CONSTANTS & CONFIGURATIONS
# ==============================================================================

RANDOM_SEED = 42
DEFAULT_SAVE_DIR = "./pdb_data"

PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}
NUCLEIC_RESIDUES = {"A", "C", "G", "U", "DA", "DC", "DG", "DT"}
ALL_RESIDUES = PROTEIN_RESIDUES | NUCLEIC_RESIDUES
RESIDUE_MAP = {res: idx + 1 for idx, res in enumerate(sorted(list(ALL_RESIDUES)))}

# PDB datasets
PDB_IDS = ["1ubq", "1pga", "1shg", "1csp", "1a70", "1crn", "2cro", "1acx"]

GRID_SIZE = 64
DEFAULT_BOX_SIZE = 32.0
DEFAULT_RADIUS = 1.5

# Mean-Shift Peak Finding Constants
DEFAULT_PEAK_THRESHOLD = 0.30
DEFAULT_PEAK_BANDWIDTH = 1.0
DEFAULT_MAX_PEAKS = 128
DEFAULT_PEAK_ITERATIONS = 5
CLASH_LIMIT = 0.6

# Pairformer Hyperparameters
EMBED_DIM_S = 64
EMBED_DIM_Z = 32
NUM_HEADS = 4
NUM_BLOCKS = 3
MAX_REL_POS = 32
CONTACT_THRESHOLD = 8.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# SECTION 1: ELEGANT RASTERIZATION & DATA PIPELINE
# ==============================================================================

def download_pdb_cif(pdb_id: str) -> str:
    path = rcsb.fetch(pdb_id, "cif", DEFAULT_SAVE_DIR)
    if isinstance(path, list):
        return str(path[0])
    return str(path)


def rasterize_structure(coords: torch.Tensor, res_indices: torch.Tensor, sigma: float = 0.8, radius: float = 0.8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized, chunk-free grid rasterization using torch.cdist.
    """
    ticks = torch.linspace(0.0, DEFAULT_BOX_SIZE, GRID_SIZE, device=coords.device)
    grid_x, grid_y, grid_z = torch.meshgrid(ticks, ticks, ticks, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)

    dists = torch.cdist(grid, coords)
    density = torch.exp(-dists**2 / (2 * sigma**2)).sum(dim=-1).view(GRID_SIZE, GRID_SIZE, GRID_SIZE)
    binary_grid = (dists <= radius).any(dim=-1).float().view(GRID_SIZE, GRID_SIZE, GRID_SIZE)

    min_dists, min_idx = torch.min(dists, dim=-1)
    residue_grid = torch.zeros(grid.shape[0], dtype=torch.long, device=coords.device)
    valid_mask = min_dists <= radius
    residue_grid[valid_mask] = res_indices[min_idx[valid_mask]]

    return density, binary_grid, residue_grid.view(GRID_SIZE, GRID_SIZE, GRID_SIZE)


def crop_and_rasterize_dynamic(structures: list, is_training: bool = False, return_coords: bool = False) -> tuple:
    coords, res_indices = random.choice(structures)
    center = coords[torch.randint(0, len(coords), (1,)).item()]

    half_box = DEFAULT_BOX_SIZE / 2.0
    mask = torch.all((coords >= center - half_box) & (coords <= center + half_box), dim=-1)

    cropped_coords = coords[mask] - center + half_box
    cropped_res = res_indices[mask]

    sigma = random.uniform(0.8, 1.8) if is_training else 1.2
    noise = random.uniform(0.01, 0.08) if is_training else 0.04

    density, binary_grid, residue_grid = rasterize_structure(cropped_coords, cropped_res, sigma=sigma, radius=DEFAULT_RADIUS)

    out_density = F.relu(density + torch.randn_like(density) * noise)
    if return_coords:
        return out_density, binary_grid, residue_grid, cropped_coords, cropped_res
    return out_density, binary_grid, residue_grid


def augment_batch_3d_joint(inputs: torch.Tensor, target_atoms: torch.Tensor, target_res: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for b in range(inputs.shape[0]):
        for dim in (-3, -2, -1):
            if random.random() > 0.5:
                inputs[b] = torch.flip(inputs[b], [dim])
                target_atoms[b] = torch.flip(target_atoms[b], [dim])
                target_res[b] = torch.flip(target_res[b], [dim])
        for plane in [(-3, -2), (-2, -1), (-3, -1)]:
            k = random.randint(0, 3)
            if k > 0:
                inputs[b] = torch.rot90(inputs[b], k, plane)
                target_atoms[b] = torch.rot90(target_atoms[b], k, plane)
                target_res[b] = torch.rot90(target_res[b], k, plane)
    return inputs, target_atoms, target_res


class BCEDiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        pred = torch.sigmoid(logits)
        p_flat, t_flat = pred.flatten(1), target.flatten(1)
        intersection = (p_flat * t_flat).sum(dim=-1)
        dice = 1.0 - (2.0 * intersection + 1e-6) / (p_flat.sum(dim=-1) + t_flat.sum(dim=-1) + 1e-6)
        return bce + dice.mean()


def find_groups(c: int) -> int:
    """Finds a divisor of c that is <= 32 to satisfy nn.GroupNorm divisibility rules."""
    for g in [32, 16, 8, 4, 2]:
        if c % g == 0: return g
    return 1


class ConvBlock3d(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(find_groups(in_c), in_c), nn.SiLU(),
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(find_groups(out_c), out_c), nn.SiLU(),
            nn.Conv3d(out_c, out_c, 3, padding=1)
        )
        self.skip = nn.Identity() if in_c == out_c else nn.Conv3d(in_c, out_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.net(x)


class SpatialTransformerBlock3d(nn.Module):
    def __init__(self, channels: int, n_heads: int):
        super().__init__()
        self.norm = nn.GroupNorm(find_groups(channels), channels)
        self.proj_in = nn.Conv3d(channels, channels, 1)
        self.proj_out = nn.Conv3d(channels, channels, 1)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels)
        )
        self.norm_attn = nn.LayerNorm(channels)
        self.norm_ff = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, d = x.shape
        h_in = x
        x = self.norm(x)
        x = self.proj_in(x).permute(0, 2, 3, 4, 1).view(b, h * w * d, c)
        x = x + self.attn(self.norm_attn(x), self.norm_attn(x), self.norm_attn(x))[0]
        x = x + self.ff(self.norm_ff(x))
        x = x.view(b, h, w, d, c).permute(0, 4, 1, 2, 3)
        return h_in + self.proj_out(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_features: int = 32) -> None:
        super().__init__()
        f = init_features
        self.down1 = ConvBlock3d(in_channels, f)
        self.pool1 = nn.Conv3d(f, f, 3, stride=2, padding=1)
        self.down2 = ConvBlock3d(f, f * 2)
        self.pool2 = nn.Conv3d(f * 2, f * 2, 3, stride=2, padding=1)

        self.bottleneck = nn.Sequential(
            ConvBlock3d(f * 2, f * 4),
            SpatialTransformerBlock3d(f * 4, n_heads=2)
        )

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv3d(f * 4, f * 4, 3, padding=1))
        self.conv_up1 = ConvBlock3d(f * 6, f * 2)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv3d(f * 2, f * 2, 3, padding=1))
        self.conv_up2 = ConvBlock3d(f * 3, f)

        self.out_conv = nn.Conv3d(f, out_channels, 1)
        self.ds_conv = nn.Conv3d(f * 2, out_channels, 1)

    def forward(self, x: torch.Tensor, return_ds: bool = False) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        x2 = self.down2(p1)
        p2 = self.pool2(x2)

        b = self.bottleneck(p2)

        u1 = self.up1(b)
        x3 = self.conv_up1(torch.cat([u1, x2], dim=1))

        u2 = self.up2(x3)
        x4 = self.conv_up2(torch.cat([u2, x1], dim=1))

        out = self.out_conv(x4)
        if return_ds:
            return out, self.ds_conv(x3)
        return out


# ==============================================================================
# SECTION 3: 3D PEAK FINDING (MEAN-SHIFT)
# ==============================================================================

class BatchedMeanShiftPeakFinder3D(nn.Module):
    def forward(self, density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = density.shape[0]
        X = density.shape[2]  # Handles 5D shape [B, 1, X, X, X]
        M = DEFAULT_MAX_PEAKS
        spacing = DEFAULT_BOX_SIZE / (X - 1)

        ticks = torch.linspace(0.0, DEFAULT_BOX_SIZE, X, device=density.device)
        grid = torch.stack(torch.meshgrid(ticks, ticks, ticks, indexing='ij'), dim=-1).view(-1, 3)

        max_pooled = F.max_pool3d(density, kernel_size=3, stride=1, padding=1)
        peaks = (density == max_pooled) & (density > DEFAULT_PEAK_THRESHOLD)

        out_coords = torch.zeros(B, M, 3, device=density.device)
        out_vals = torch.zeros(B, M, device=density.device)
        out_mask = torch.zeros(B, M, dtype=torch.bool, device=density.device)

        clash_limit_sq = CLASH_LIMIT ** 2
        inv_two_bandwidth_sq = 1.0 / (2 * DEFAULT_PEAK_BANDWIDTH ** 2)

        # ----------------------------------------------------------------------
        # 1. Dynamic Batched Active Coordinate Padding
        # ----------------------------------------------------------------------
        active_mask = (density > DEFAULT_PEAK_THRESHOLD).view(B, -1)
        num_active = active_mask.sum(dim=-1)
        max_active = num_active.max().item()

        # ----------------------------------------------------------------------
        # 2. Dynamic Batched Seed Padding
        # ----------------------------------------------------------------------
        peaks_mask = peaks.view(B, -1)
        cum_sum_peaks = torch.cumsum(peaks_mask, dim=-1)
        peaks_positions = (cum_sum_peaks * peaks_mask) - 1
        
        valid_peaks_mask = peaks_mask & (peaks_positions < M)
        num_peaks = valid_peaks_mask.sum(dim=-1)
        S_max = num_peaks.max().item()

        if max_active == 0 or S_max == 0:
            return out_coords, out_vals, out_mask

        # Populate padded active tensors
        nz_active = torch.nonzero(active_mask)
        pos_active = ((torch.cumsum(active_mask, dim=-1) * active_mask) - 1)[active_mask]
        
        padded_active_coords = torch.zeros(B, max_active, 3, device=density.device)
        padded_active_weights = torch.zeros(B, max_active, device=density.device)
        padded_active_coords[nz_active[:, 0], pos_active] = grid[nz_active[:, 1]]
        padded_active_weights[nz_active[:, 0], pos_active] = density.view(B, -1)[active_mask]

        # Populate padded seeds
        nz_peaks = torch.nonzero(valid_peaks_mask)
        pos_peaks = peaks_positions[valid_peaks_mask]
        
        padded_seeds = torch.zeros(B, S_max, 3, device=density.device)
        padded_seeds[nz_peaks[:, 0], pos_peaks] = grid[nz_peaks[:, 1]]

        # Keep track of active seed masks per batch element
        valid_seed_mask = torch.arange(S_max, device=density.device).unsqueeze(0) < num_peaks.unsqueeze(1)
        keep = valid_seed_mask.clone()

        # ----------------------------------------------------------------------
        # 3. Batched Mean-Shift Iteration (No Batch Loop)
        # ----------------------------------------------------------------------
        padded_active_sq = torch.sum(padded_active_coords ** 2, dim=-1, keepdim=True).transpose(-1, -2)
        for _ in range(DEFAULT_PEAK_ITERATIONS):
            seeds_sq = torch.sum(padded_seeds ** 2, dim=-1, keepdim=True)
            dot_product = torch.bmm(padded_seeds, padded_active_coords.transpose(-1, -2))
            sq_dists = torch.clamp(seeds_sq + padded_active_sq - 2 * dot_product, min=0.0)

            weights = torch.exp(-sq_dists * inv_two_bandwidth_sq) * padded_active_weights.unsqueeze(1)
            padded_seeds = torch.bmm(weights, padded_active_coords) / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # ----------------------------------------------------------------------
        # 4. Batched Clash Removal (Parallelized across Batch elements)
        # ----------------------------------------------------------------------
        for i in range(S_max):
            if i == S_max - 1: break
            seeds_i = padded_seeds[:, i:i+1]
            seeds_rest = padded_seeds[:, i+1:]
            
            sq_dists = torch.sum((seeds_rest - seeds_i) ** 2, dim=-1)
            clash = (sq_dists < clash_limit_sq) & keep[:, i:i+1]
            keep[:, i+1:][clash] = False

        # ----------------------------------------------------------------------
        # 5. Pack Kept Seeds to Contiguous Output Tensors
        # ----------------------------------------------------------------------
        keep_cumsum = torch.cumsum(keep, dim=-1)
        keep_positions = (keep_cumsum * keep) - 1
        final_keep = keep & (keep_positions < M)

        nz_final = torch.nonzero(final_keep)
        b_final = nz_final[:, 0]
        s_final = nz_final[:, 1]
        pos_final = keep_positions[final_keep]

        coords_final = padded_seeds[b_final, s_final]
        coords_grid = (coords_final / spacing).round().long().clamp(0, X-1)

        out_coords[b_final, pos_final] = coords_final
        out_vals[b_final, pos_final] = density[b_final, 0, coords_grid[:, 0], coords_grid[:, 1], coords_grid[:, 2]]
        out_mask[b_final, pos_final] = True

        return out_coords, out_vals, out_mask


# ==============================================================================
# SECTION 4: THE PAIRFORMER ARCHITECTURE (Lecture 3)
# ==============================================================================

class RelativePositionEmbedding(nn.Module):
    def __init__(self, max_rel_pos: int = 32, c_z: int = 32):
        super().__init__()
        self.max_rel_pos = max_rel_pos
        self.num_bins = 2 * max_rel_pos + 1
        self.emb = nn.Embedding(self.num_bins, c_z)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        return self.emb(torch.clamp(diff, -self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos)


class SequenceToPairInitializer(nn.Module):
    def __init__(self, c_s: int, c_z: int, max_rel_pos: int = 32):
        super().__init__()
        self.linear_s_q = nn.Linear(c_s, c_z)
        self.linear_s_k = nn.Linear(c_s, c_z)
        self.rel_pos = RelativePositionEmbedding(max_rel_pos, c_z)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s_q = self.linear_s_q(s).unsqueeze(2)
        s_k = self.linear_s_k(s).unsqueeze(1)
        return s_q + s_k + self.rel_pos(s.shape[1], s.device).unsqueeze(0)


class AttentionPairBias(nn.Module):
    def __init__(self, c_s: int, c_z: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.c_s = c_s
        self.d_k = c_s // n_heads
        self.proj_q = nn.Linear(c_s, c_s, bias=False)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_bias = nn.Linear(c_z, n_heads, bias=False)
        self.proj_gate = nn.Linear(c_s, c_s)
        self.proj_out = nn.Linear(c_s, c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor, shape_watch: bool = False) -> torch.Tensor:
        B, N, _ = s.shape
        H = self.n_heads

        q = self.proj_q(s).view(B, N, H, self.d_k).permute(0, 2, 1, 3)
        k = self.proj_k(s).view(B, N, H, self.d_k).permute(0, 2, 1, 3)
        v = self.proj_v(s).view(B, N, H, self.d_k).permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.d_k ** 0.5)
        bias = self.proj_bias(z).permute(0, 3, 1, 2)

        attn_probs = F.softmax(logits + bias, dim=-1)
        out = torch.matmul(attn_probs, v)

        gate = torch.sigmoid(self.proj_gate(s).view(B, N, H, self.d_k).permute(0, 2, 1, 3))
        out = (out * gate).permute(0, 2, 1, 3).contiguous().view(B, N, self.c_s)

        if shape_watch:
            print(f"    [Shape Watcher - AttentionPairBias]\n      s={list(s.shape)}, z={list(z.shape)}, Q/K/V heads={list(q.shape)}")

        return self.proj_out(out)


class OuterProductMean(nn.Module):
    def __init__(self, c_s: int, c_z: int, c_hidden: int = 16):
        super().__init__()
        self.ln = nn.LayerNorm(c_s)
        self.lin1 = nn.Linear(c_s, c_hidden)
        self.lin2 = nn.Linear(c_s, c_hidden)
        self.lin_out = nn.Linear(c_hidden ** 2, c_z)

    def forward(self, s: torch.Tensor, shape_watch: bool = False) -> torch.Tensor:
        s_norm = self.ln(s)
        a = self.lin1(s_norm)
        b = self.lin2(s_norm)
        outer = torch.einsum("b i c, b j d -> b i j c d", a, b).flatten(start_dim=-2)
        out = self.lin_out(outer)

        if shape_watch:
            print(f"    [Shape Watcher - OuterProductMean]\n      outer_flat={list(outer.shape)}, z_update={list(out.shape)}")

        return out


class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, c_z: int, c_hidden: int = 32):
        super().__init__()
        self.ln = nn.LayerNorm(c_z)
        self.lin_a = nn.Linear(c_z, c_hidden)
        self.lin_b = nn.Linear(c_z, c_hidden)
        self.lin_gate_a = nn.Linear(c_z, c_hidden)
        self.lin_gate_b = nn.Linear(c_z, c_hidden)
        self.lin_out = nn.Linear(c_hidden, c_z)
        self.lin_gate_out = nn.Linear(c_z, c_z)

    def forward(self, z: torch.Tensor, shape_watch: bool = False) -> torch.Tensor:
        z_norm = self.ln(z)
        a = torch.sigmoid(self.lin_gate_a(z_norm)) * self.lin_a(z_norm)
        b = torch.sigmoid(self.lin_gate_b(z_norm)) * self.lin_b(z_norm)
        out = torch.einsum("b i k c, b j k c -> b i j c", a, b)
        final_out = self.lin_out(out) * torch.sigmoid(self.lin_gate_out(z_norm))

        if shape_watch:
            print(f"    [Shape Watcher - TriangleMultiplicativeUpdate]\n      gathered_sum={list(out.shape)}")

        return final_out


class PairformerBlock(nn.Module):
    def __init__(self, c_s: int, c_z: int, n_heads: int = 4):
        super().__init__()
        self.attn = AttentionPairBias(c_s, c_z, n_heads)
        self.ln_s1 = nn.LayerNorm(c_s)
        self.transition_s = nn.Sequential(
            nn.LayerNorm(c_s), nn.Linear(c_s, 4 * c_s), nn.GELU(), nn.Linear(4 * c_s, c_s)
        )
        self.opm = OuterProductMean(c_s, c_z)
        self.tri_mul = TriangleMultiplicativeUpdate(c_z)
        self.transition_z = nn.Sequential(
            nn.LayerNorm(c_z), nn.Linear(c_z, 4 * c_z), nn.GELU(), nn.Linear(4 * c_z, c_z)
        )

    def forward(self, s: torch.Tensor, z: torch.Tensor, shape_watch: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if shape_watch: print("\n=== STARTING PAIRFORMER BLOCK SHAPE-WATCHING ===")
        s = s + self.attn(self.ln_s1(s), z, shape_watch=shape_watch)
        s = s + self.transition_s(s)
        z = z + self.opm(s, shape_watch=shape_watch)
        z = z + self.tri_mul(z, shape_watch=shape_watch)
        z = z + self.transition_z(z)
        if shape_watch: print("=== END PAIRFORMER BLOCK SHAPE-WATCHING ===\n")
        return s, z


class PairformerStack(nn.Module):
    def __init__(self, c_s: int, c_z: int, n_blocks: int = 3, n_heads: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([PairformerBlock(c_s, c_z, n_heads) for _ in range(n_blocks)])

    def forward(self, s: torch.Tensor, z: torch.Tensor, shape_watch_first: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        for i, block in enumerate(self.blocks):
            s, z = block(s, z, shape_watch=(shape_watch_first and i == 0))
        return s, z


class PairformerContactPredictor(nn.Module):
    def __init__(self, vocab_size: int, c_s: int = 64, c_z: int = 32, n_blocks: int = 3, n_heads: int = 4, max_rel_pos: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, c_s, padding_idx=0)
        self.initializer = SequenceToPairInitializer(c_s, c_z, max_rel_pos)
        self.pairformer = PairformerStack(c_s, c_z, n_blocks, n_heads)
        self.contact_head = nn.Sequential(
            nn.LayerNorm(c_z), nn.Linear(c_z, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor, shape_watch: bool = False) -> torch.Tensor:
        s = self.embedding(x)
        z = self.initializer(s)
        s, z = self.pairformer(s, z, shape_watch_first=shape_watch)
        return self.contact_head(z).squeeze(-1)


def print_ascii_contact_map(gt: torch.Tensor, pred: torch.Tensor, threshold: float = 0.5):
    N = gt.shape[0]
    step = max(1, N // 35)
    print("\n   GROUND TRUTH CONTACT MAP" + " " * 18 + "PREDICTED CONTACT MAP")
    print("   " + "-" * (N // step) + "      " + "-" * (N // step))
    for i in range(0, N, step):
        row_gt = "".join("#" if gt[i, j] else "." for j in range(0, N, step))
        row_pred = "".join("#" if pred[i, j] > threshold else "." for j in range(0, N, step))
        print(f"{i:2d} {row_gt}      {i:2d} {row_pred}")
    print("   " + "-" * (N // step) + "      " + "-" * (N // step) + "\n")


# ==============================================================================
# SECTION 4B: THE EM-PAIRFORMER ARCHITECTURE (CryoZeta)
# ==============================================================================

class InterMultiplicativeUpdate(nn.Module):
    def __init__(self, c_z: int, c_p: int, c_hidden: int, c_pz: int, outgoing: bool = True):
        super().__init__()
        self.outgoing = outgoing
        self.ln_p = nn.LayerNorm(c_p)
        self.ln_z = nn.LayerNorm(c_z)
        self.ln_out = nn.LayerNorm(c_hidden)

        self.lin_p_proj = nn.Linear(c_p, c_hidden)
        self.lin_z_proj = nn.Linear(c_z, c_hidden)
        self.lin_p_gate = nn.Linear(c_p, c_hidden)
        self.lin_z_gate = nn.Linear(c_z, c_hidden)
        self.lin_out = nn.Linear(c_hidden, c_pz)

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # z: [B, N_res, N_res, c_z], p: [B, N_point, N_point, c_p]
        p_norm = self.ln_p(p)
        z_norm = self.ln_z(z)

        x_p = torch.sigmoid(self.lin_p_gate(p_norm)) * self.lin_p_proj(p_norm)
        x_z = torch.sigmoid(self.lin_z_gate(z_norm)) * self.lin_z_proj(z_norm)

        if self.outgoing:
            # sum over intermediate connections to find outgoing paths (dim -2)
            z_p = x_p.sum(dim=-2) # [B, N_point, c_hidden]
            z_z = x_z.sum(dim=-2) # [B, N_res, c_hidden]
        else:
            # incoming paths (dim -3)
            z_p = x_p.sum(dim=-3) # [B, N_point, c_hidden]
            z_z = x_z.sum(dim=-3) # [B, N_res, c_hidden]

        out = torch.einsum("b p h, b r h -> b p r h", z_p, z_z)
        return self.lin_out(self.ln_out(out))


class JointAttentionWithPairBias(nn.Module):
    def __init__(self, c_pz: int, c_bias: int, c_hidden: int, n_heads: int = 4, along_dim: int = -2):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = c_hidden // n_heads
        self.along_dim = along_dim  # -2: row (residue), -3: column (point)
        self.ln_pz = nn.LayerNorm(c_pz)
        self.ln_bias = nn.LayerNorm(c_bias)

        self.proj_q = nn.Linear(c_pz, c_hidden, bias=False)
        self.proj_k = nn.Linear(c_pz, c_hidden, bias=False)
        self.proj_v = nn.Linear(c_pz, c_hidden, bias=False)
        self.proj_bias = nn.Linear(c_bias, n_heads, bias=False)
        self.proj_gate = nn.Linear(c_pz, c_hidden)
        self.proj_out = nn.Linear(c_hidden, c_pz)

    def forward(self, pz: torch.Tensor, bias_tensor: torch.Tensor) -> torch.Tensor:
        B, P, R, _ = pz.shape
        H = self.n_heads
        pz_norm = self.ln_pz(pz)

        perm = (0, 1, 3, 2, 4) if self.along_dim == -2 else (0, 2, 3, 1, 4)
        perm_out = (0, 1, 3, 2, 4) if self.along_dim == -2 else (0, 3, 1, 2, 4)

        q = self.proj_q(pz_norm).view(B, P, R, H, self.d_k).permute(*perm)
        k = self.proj_k(pz_norm).view(B, P, R, H, self.d_k).permute(*perm)
        v = self.proj_v(pz_norm).view(B, P, R, H, self.d_k).permute(*perm)

        bias = self.proj_bias(self.ln_bias(bias_tensor)).permute(0, 3, 1, 2).unsqueeze(1)
        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.d_k ** 0.5)
        attn_probs = F.softmax(logits + bias, dim=-1)

        out = torch.matmul(attn_probs, v)
        gate = torch.sigmoid(self.proj_gate(pz_norm).view(B, P, R, H, self.d_k).permute(*perm))

        out = (out * gate).permute(*perm_out).contiguous().view(B, P, R, -1)
        return self.proj_out(out)


class PointResidueTransition(nn.Module):
    def __init__(self, c_pz: int, n: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(c_pz),
            nn.Linear(c_pz, n * c_pz),
            nn.ReLU(),
            nn.Linear(n * c_pz, c_pz)
        )

    def forward(self, pz: torch.Tensor) -> torch.Tensor:
        return self.net(pz)


class EMOuterProductMean(nn.Module):
    def __init__(self, c_m: int, c_z: int, c_hidden: int = 16):
        super().__init__()
        self.ln = nn.LayerNorm(c_m)
        self.lin1 = nn.Linear(c_m, c_hidden)
        self.lin2 = nn.Linear(c_m, c_hidden)
        self.lin_out = nn.Linear(c_hidden ** 2, c_z)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        m_norm = self.ln(m)
        a = self.lin1(m_norm)
        b = self.lin2(m_norm)
        outer = torch.einsum("b s i c, b s j d -> b i j c d", a, b).flatten(start_dim=-2)
        out = self.lin_out(outer) / (m.shape[1] + 1e-8)
        return out


class EMPairformerBlock(nn.Module):
    def __init__(self, c_s: int, c_z: int, c_pz: int, c_p: int, n_heads: int = 4):
        super().__init__()
        self.tri_mul_out = TriangleMultiplicativeUpdate(c_z)
        self.tri_mul_in = TriangleMultiplicativeUpdate(c_z)

        self.inter_outgoing = InterMultiplicativeUpdate(c_z, c_p, c_hidden=c_pz, c_pz=c_pz, outgoing=True)
        self.inter_incoming = InterMultiplicativeUpdate(c_z, c_p, c_hidden=c_pz, c_pz=c_pz, outgoing=False)

        self.residue_row_attn = JointAttentionWithPairBias(c_pz, c_z, c_hidden=c_pz, n_heads=n_heads, along_dim=-2)
        self.point_column_attn = JointAttentionWithPairBias(c_pz, c_p, c_hidden=c_pz, n_heads=n_heads, along_dim=-3)
        self.pz_transition = PointResidueTransition(c_pz)

        self.outer_row_opm = EMOuterProductMean(c_pz, c_z, c_hidden=16)
        self.outer_col_opm = EMOuterProductMean(c_pz, c_p, c_hidden=16)

        self.transition_z = nn.Sequential(
            nn.LayerNorm(c_z), nn.Linear(c_z, 4 * c_z), nn.GELU(), nn.Linear(4 * c_z, c_z)
        )
        self.c_s = c_s
        if c_s > 0:
            self.attn = AttentionPairBias(c_s, c_z, n_heads)
            self.transition_s = nn.Sequential(
                nn.LayerNorm(c_s), nn.Linear(c_s, 4 * c_s), nn.GELU(), nn.Linear(4 * c_s, c_s)
            )

    def forward(self, s: torch.Tensor | None, z: torch.Tensor, pz: torch.Tensor, p: torch.Tensor, shape_watch: bool = False) -> tuple:
        if shape_watch:
            print(f"    [Shape Watcher - EMPairformerBlock Input]\n      s={list(s.shape) if s is not None else None}, z={list(z.shape)}, pz={list(pz.shape)}, p={list(p.shape)}")

        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)

        pz = pz + self.inter_outgoing(z, p)
        pz = pz + self.inter_incoming(z, p)
        pz = pz + self.residue_row_attn(pz, z)
        pz = pz + self.point_column_attn(pz, p)
        pz = pz + self.pz_transition(pz)

        # update z from pz (averaging over point axis)
        z = z + self.outer_row_opm(pz)
        # update p from pz (averaging over residue axis)
        p = p + self.outer_col_opm(pz.transpose(1, 2))
        z = z + self.transition_z(z)

        if self.c_s > 0 and s is not None:
            s = s + self.attn(s, z, shape_watch=shape_watch)
            s = s + self.transition_s(s)

        return s, z, pz, p


class EMPairformerStack(nn.Module):
    def __init__(self, c_s: int, c_z: int, c_pz: int, c_p: int, n_blocks: int = 3, n_heads: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([EMPairformerBlock(c_s, c_z, c_pz, c_p, n_heads) for _ in range(n_blocks)])

    def forward(self, s: torch.Tensor | None, z: torch.Tensor, pz: torch.Tensor, p: torch.Tensor, shape_watch_first: bool = False) -> tuple:
        for i, block in enumerate(self.blocks):
            s, z, pz, p = block(s, z, pz, p, shape_watch=(shape_watch_first and i == 0))
        return s, z, pz, p


# ==============================================================================
# SECTION 5: K-FOLD AND DUAL GOOGLE COLAB TRAINING PIPELINE
# ==============================================================================

def get_k_folds(items: list, k: int = 5, seed: int = 42) -> list[list]:
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
    val_idx = (fold_idx + 1) % k

    test_set = folds[test_idx]
    val_set = folds[val_idx]

    train_set = []
    for idx in range(k):
        if idx != test_idx and idx != val_idx:
            train_set.extend(folds[idx])

    return train_set, val_set, test_set


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("===========================================================================")
    print(" PREPARING MOLECULAR DATABASE ")
    print("===========================================================================")

    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
    all_structures = {}
    for pid in tqdm(PDB_IDS, desc="Caching mmCIF files"):
        filepath = download_pdb_cif(pid)
        try:
            atoms = pdbx.get_structure(pdbx.CIFFile.read(filepath), model=1)
            # 1. 3D Volumetric representations (All atoms)
            valid_atoms = atoms[np.isin(atoms.res_name, list(ALL_RESIDUES))]
            v_coords = torch.tensor(valid_atoms.coord, dtype=torch.float32)
            v_res_idx = torch.tensor([RESIDUE_MAP[name] for name in valid_atoms.res_name], dtype=torch.long)

            # 2. 1D Sequential representations (C-alpha deduplicated)
            rep_atoms = atoms[(atoms.atom_name == "CA") | (atoms.atom_name == "C4'") | (atoms.atom_name == "P")]
            seen = set()
            filtered = []
            for idx in range(len(rep_atoms)):
                res_key = (rep_atoms.chain_id[idx], rep_atoms.res_id[idx])
                if res_key not in seen:
                    seen.add(res_key)
                    filtered.append(idx)
            rep_atoms = rep_atoms[filtered]
            s_seq = torch.tensor([RESIDUE_MAP.get(name, 0) for name in rep_atoms.res_name], dtype=torch.long)
            s_coords = torch.tensor(rep_atoms.coord, dtype=torch.float32)

            all_structures[pid] = {
                "v_coords": v_coords, "v_res_idx": v_res_idx,
                "s_seq": s_seq, "s_coords": s_coords
            }
        except Exception as e:
            print(f"Failed to load PDB {pid}: {e}")

    # --------------------------------------------------------------------------
    # DEMO 1: 5-FOLD 3D VOLUMETRIC U-NET CROSS-VALIDATION (LECTURE 2)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" PIPELINE 1: TRAINING 5-FOLD VOLUMETRIC U-NET DEMO ")
    print("="*70)

    K_FOLDS = 5
    NUM_EPOCHS = 1 if device.type == "cpu" else 100
    steps_per_epoch = 1 if device.type == "cpu" else 25
    MATCHING_RADIUS = 1.5
    spacing = DEFAULT_BOX_SIZE / (GRID_SIZE - 1)

    # Initialize 5 folds on the 8 PDBs
    pdb_folds = get_k_folds(PDB_IDS, k=K_FOLDS, seed=RANDOM_SEED)
    all_folds_pdb_results = []

    global_gt_atoms = 0
    global_matched_atoms = 0
    global_correct_residues = 0
    global_resolved_peaks = 0
    global_num_crops = 0

    for fold_idx in range(K_FOLDS):
        print("\n" + "="*70)
        print(f" RUNNING FOLD {fold_idx + 1}/{K_FOLDS} (60/20/20 SPLIT) ")
        print("="*70)

        train_pids, val_pids, test_pids = get_fold_split(pdb_folds, fold_idx)
        print(f"Train structures ({len(train_pids)}): {[p.upper() for p in train_pids]}")
        print(f"Validation structures ({len(val_pids)}): {[p.upper() for p in val_pids]}")
        print(f"Test structures ({len(test_pids)}): {[p.upper() for p in test_pids]}")

        # Construct splits
        v_train = [(all_structures[pid]["v_coords"], all_structures[pid]["v_res_idx"]) for pid in train_pids]
        v_val = [(all_structures[pid]["v_coords"], all_structures[pid]["v_res_idx"]) for pid in val_pids]
        v_test = [(all_structures[pid]["v_coords"], all_structures[pid]["v_res_idx"]) for pid in test_pids]

        # Initialize models
        unet_atom = UNet3D(1, 1, init_features=16).to(device)
        unet_res = UNet3D(1, len(RESIDUE_MAP) + 1, init_features=16).to(device)
        opt_unet = torch.optim.Adam(list(unet_atom.parameters()) + list(unet_res.parameters()), lr=0.001)
        criterion_atom = BCEDiceLoss()
        criterion_res = nn.CrossEntropyLoss(ignore_index=0)

        def compute_unet_loss(pred_atom, ds_atom, target_atoms, pred_res, ds_res, target_res):
            loss_atom_main = criterion_atom(pred_atom, target_atoms)
            loss_res_main = criterion_res(pred_res, target_res)
            target_atoms_ds = F.max_pool3d(target_atoms, kernel_size=2, stride=2)
            target_res_ds = F.max_pool3d(target_res.float().unsqueeze(1), kernel_size=2, stride=2).squeeze(1).long()
            loss_atom_ds = criterion_atom(ds_atom, target_atoms_ds)
            loss_res_ds = criterion_res(ds_res, target_res_ds)
            return loss_atom_main + loss_res_main + 0.5 * (loss_atom_ds + loss_res_ds)

        best_val_loss = float('inf')
        best_atom_state = None
        best_res_state = None

        # Train loop
        for epoch in range(1, NUM_EPOCHS + 1):
            unet_atom.train(); unet_res.train()
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                samples = [crop_and_rasterize_dynamic(v_train, is_training=True) for _ in range(4)]
                inputs = torch.stack([s[0] for s in samples]).unsqueeze(1).to(device)
                target_atoms = torch.stack([s[1] for s in samples]).unsqueeze(1).to(device)
                target_res = torch.stack([s[2] for s in samples]).long().to(device)

                inputs, target_atoms, target_res = augment_batch_3d_joint(inputs, target_atoms, target_res)

                opt_unet.zero_grad()
                pred_atom, ds_atom = unet_atom(inputs, return_ds=True)
                pred_res, ds_res = unet_res(inputs, return_ds=True)

                loss = compute_unet_loss(pred_atom, ds_atom, target_atoms, pred_res, ds_res, target_res)
                loss.backward()
                opt_unet.step()
                epoch_loss += loss.item()
            epoch_loss /= steps_per_epoch

            # Periodic evaluation & clean printing
            if epoch % 50 == 0 or epoch == 1:
                unet_atom.eval(); unet_res.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_steps = 1 if device.type == "cpu" else 5
                    for _ in range(val_steps):
                        val_samples = [crop_and_rasterize_dynamic(v_val, is_training=False) for _ in range(4)]
                        val_inputs = torch.stack([s[0] for s in val_samples]).unsqueeze(1).to(device)
                        val_target_atoms = torch.stack([s[1] for s in val_samples]).unsqueeze(1).to(device)
                        val_target_res = torch.stack([s[2] for s in val_samples]).long().to(device)

                        val_pred_atom, val_ds_atom = unet_atom(val_inputs, return_ds=True)
                        val_pred_res, val_ds_res = unet_res(val_inputs, return_ds=True)
                        val_loss += compute_unet_loss(val_pred_atom, val_ds_atom, val_target_atoms, val_pred_res, val_ds_res, val_target_res).item()
                    val_loss /= val_steps

                print(f"Fold {fold_idx + 1} | Epoch {epoch:03d}/100 | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_atom_state = {k: v.cpu() for k, v in unet_atom.state_dict().items()}
                    best_res_state = {k: v.cpu() for k, v in unet_res.state_dict().items()}

        # Restore best model for testing
        if best_atom_state is not None:
            unet_atom.load_state_dict({k: v.to(device) for k, v in best_atom_state.items()})
            unet_res.load_state_dict({k: v.to(device) for k, v in best_res_state.items()})

        # Test Evaluation for this fold
        print(f"\nEvaluating Fold {fold_idx + 1} on unseen test structures...")
        unet_atom.eval(); unet_res.eval()
        peak_finder = BatchedMeanShiftPeakFinder3D().to(device)

        fold_pdb_results = []
        num_test_crops = 1 if device.type == "cpu" else 10  # 10 random crops per unseen test target

        for pid in test_pids:
            # Single PDB subset
            test_target_structure = [(all_structures[pid]["v_coords"], all_structures[pid]["v_res_idx"])]
            pid_gt_atoms, pid_matched_atoms, pid_correct_residues, pid_resolved_peaks = 0, 0, 0, 0

            with torch.no_grad():
                for _ in range(num_test_crops):
                    test_input, _, _, gt_coords, gt_res_indices = crop_and_rasterize_dynamic(
                        test_target_structure, is_training=False, return_coords=True
                    )
                    test_in_batch = test_input.unsqueeze(0).unsqueeze(0).to(device)

                    pred_density = F.relu(unet_atom(test_in_batch))
                    pred_coords, _, pred_mask = peak_finder(pred_density)
                    pred_res_logits = unet_res(test_in_batch)

                    pred_coords = pred_coords[0].cpu()
                    num_pred_peaks = pred_mask[0].sum().item()

                    pid_resolved_peaks += num_pred_peaks
                    pid_gt_atoms += len(gt_coords)

                    if num_pred_peaks > 0 and len(gt_coords) > 0:
                        dists = torch.cdist(gt_coords.cpu(), pred_coords[:num_pred_peaks])
                        min_dists, min_idxs = torch.min(dists, dim=-1)
                        for j, (dist, idx) in enumerate(zip(min_dists, min_idxs)):
                            if dist.item() <= MATCHING_RADIUS:
                                pid_matched_atoms += 1
                                p_coord = pred_coords[idx.item()]
                                grid_idx = torch.clamp(torch.round(p_coord / spacing).long(), 0, GRID_SIZE - 1)
                                logits = pred_res_logits[0, :, grid_idx[0], grid_idx[1], grid_idx[2]]
                                if torch.argmax(logits[1:]).item() + 1 == gt_res_indices[j].item():
                                    pid_correct_residues += 1

            avg_peaks_per_crop = pid_resolved_peaks / num_test_crops
            recovery_pct = (pid_matched_atoms / pid_gt_atoms) * 100 if pid_gt_atoms > 0 else 0.0
            class_pct = (pid_correct_residues / pid_matched_atoms) * 100 if pid_matched_atoms > 0 else 0.0

            print(f"  Target {pid.upper()} | Peaks/Crop: {avg_peaks_per_crop:.1f} | Recovery: {recovery_pct:.1f}% | Classification: {class_pct:.1f}%")

            result_entry = {
                "fold": fold_idx + 1, "pid": pid.upper(), "avg_peaks": avg_peaks_per_crop,
                "gt_atoms": pid_gt_atoms, "matched_atoms": pid_matched_atoms,
                "recovery_pct": recovery_pct, "class_pct": class_pct
            }
            fold_pdb_results.append(result_entry)
            all_folds_pdb_results.append(result_entry)

            global_gt_atoms += pid_gt_atoms
            global_matched_atoms += pid_matched_atoms
            global_correct_residues += pid_correct_residues
            global_resolved_peaks += pid_resolved_peaks
            global_num_crops += num_test_crops

        # Free VRAM memory to prevent leaks on Colab
        del unet_atom, unet_res, opt_unet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # FINAL CONSOLIDATED CROSS-VALIDATION REPORT
    # --------------------------------------------------------------------------
    print("\n" + "="*85)
    print(" FINAL 5-FOLD CROSS-VALIDATION SUMMARY TABLE ")
    print("="*85)
    print(" Fold | PDB ID  | Resolved/Crop | Total Atoms | Recovery % | Classification %")
    print("-" * 85)
    for res in all_folds_pdb_results:
        print(f"  {res['fold']:<3} | {res['pid']:<7} | {res['avg_peaks']:<13.1f} | {res['gt_atoms']:<11} | {res['recovery_pct']:<10.1f}% | {res['class_pct']:<16.1f}%")
    print("-" * 85)

    global_avg_peaks = global_resolved_peaks / global_num_crops if global_num_crops > 0 else 0.0
    global_recovery = (global_matched_atoms / global_gt_atoms) * 100 if global_gt_atoms > 0 else 0.0
    global_classification = (global_correct_residues / global_matched_atoms) * 100 if global_matched_atoms > 0 else 0.0

    print(f" {'OVERALL':<7} | {'ALL':<7} | {global_avg_peaks:<13.1f} | {global_gt_atoms:<11} | {global_recovery:<10.1f}% | {global_classification:<16.1f}%")
    print("="*85 + "\n")

    # --------------------------------------------------------------------------
    # DEMO 2: PAIRFORMER SEQUENCE-TO-PAIR OPTIMIZATION (LECTURE 3)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" PIPELINE 2: TRAINING PAIRFORMER CONTACT MAP OPTIMIZATION ")
    print("="*70)

    pairformer = PairformerContactPredictor(
        vocab_size=len(RESIDUE_MAP), c_s=EMBED_DIM_S, c_z=EMBED_DIM_Z, n_blocks=NUM_BLOCKS, n_heads=NUM_HEADS
    ).to(device)
    opt_pf = torch.optim.Adam(pairformer.parameters(), lr=0.002)
    criterion_pf = nn.BCEWithLogitsLoss()

    # Precompute static target contacts to optimize speed and reduce lines
    pf_dataset = []
    for pid, s in all_structures.items():
        tokens = s["s_seq"].unsqueeze(0).to(device)
        coords = s["s_coords"].to(device)
        target = (torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0) < CONTACT_THRESHOLD).float().unsqueeze(0)
        pf_dataset.append((tokens, target))
    pf_train = pf_dataset[:-2]
    pf_val = pf_dataset[-2:]

    # Train 200 epochs on contact maps (scaled down on CPU)
    num_pf_epochs = 2 if device.type == "cpu" else 200
    for epoch in range(1, num_pf_epochs + 1):
        pairformer.train()
        train_loss = 0.0
        for tokens, target in pf_train:
            opt_pf.zero_grad()
            loss = criterion_pf(pairformer(tokens), target)
            loss.backward()
            opt_pf.step()
            train_loss += loss.item()
        train_loss /= len(pf_train)

        # Periodic evaluation & clean printing
        if epoch % 50 == 0 or epoch == 1:
            pairformer.eval()
            with torch.no_grad():
                val_loss = sum(criterion_pf(pairformer(tok), tar).item() for tok, tar in pf_val) / len(pf_val)
            print(f"Pairformer Epoch {epoch:03d}/200 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Visual check on the first target with shape-watching enabled
    pairformer.eval()
    with torch.no_grad():
        vis_key = list(all_structures.keys())[0]
        vis_s = all_structures[vis_key]
        vis_tokens = vis_s["s_seq"].unsqueeze(0).to(device)
        vis_pred = torch.sigmoid(pairformer(vis_tokens, shape_watch=True)).squeeze(0).cpu()
        vis_gt = (torch.cdist(vis_s["s_coords"].unsqueeze(0), vis_s["s_coords"].unsqueeze(0)).squeeze(0) < CONTACT_THRESHOLD).float()
        print(f"\nCompleted run. Contact Map ASCII Visual check for {vis_key.upper()}:")
        print_ascii_contact_map(vis_gt, vis_pred, threshold=0.5)
    print("\n" + "="*70)
    print(" PIPELINE 3: MATHEMATICAL DYNAMIC SHAPE WATCHING OF CRYOMAP EM-PAIRFORMER ")
    print("="*70)

    # Shape watcher unit tests for the EM-Pairformer (The Karpathy Touch!)
    def run_em_pairformer_shape_watcher_demo():
        print("\n[Shape Watcher] Initializing EM-Pairformer Demo...")
        B, N_res, N_point = 1, 20, 15
        c_s, c_z, c_pz, c_p = 64, 32, 32, 32

        # 1. Mock inputs matching physical coordinates and densities
        vocab_size = len(RESIDUE_MAP)
        mock_tokens = torch.randint(1, vocab_size, (B, N_res), device=device)
        mock_p_coords = torch.rand(B, N_point, 3, device=device) * DEFAULT_BOX_SIZE
        mock_p_densities = torch.rand(B, N_point, device=device)
        mock_res_coords = torch.rand(B, N_res, 3, device=device) * DEFAULT_BOX_SIZE

        # 2. Build physical representation matrices using RBF
        def mock_rbf_expansion(dists: torch.Tensor, num_centers: int = 32) -> torch.Tensor:
            centers = torch.linspace(0.0, DEFAULT_BOX_SIZE, num_centers, device=dists.device)
            widths = DEFAULT_BOX_SIZE / num_centers
            return torch.exp(-((dists.unsqueeze(-1) - centers) / widths) ** 2)

        print("  Generating geometric representation matrices...")
        dist_pp = torch.cdist(mock_p_coords, mock_p_coords)
        dist_pr = torch.cdist(mock_p_coords, mock_res_coords)

        p_geom = mock_rbf_expansion(dist_pp, num_centers=c_p)
        pz_geom = mock_rbf_expansion(dist_pr, num_centers=c_pz)

        # Inject density features into point representations
        proj_density = nn.Linear(1, c_p).to(device)
        p = p_geom + proj_density(mock_p_densities.unsqueeze(-1)).unsqueeze(1)
        pz = pz_geom

        # Embed sequence and initialize residue pairs
        embedding = nn.Embedding(vocab_size + 1, c_s, padding_idx=0).to(device)
        initializer = SequenceToPairInitializer(c_s, c_z, MAX_REL_POS).to(device)

        s = embedding(mock_tokens)
        z = initializer(s)

        # 3. Instantiate EMPairformerStack
        print("  Instantiating EMPairformerStack...")
        em_pairformer = EMPairformerStack(c_s=c_s, c_z=c_z, c_pz=c_pz, c_p=c_p, n_blocks=2, n_heads=4).to(device)
        em_pairformer.eval()

        # 4. Forward pass under dynamic Shape Watcher
        print("  Running EM-Pairformer forward pass:")
        with torch.no_grad():
            s_out, z_out, pz_out, p_out = em_pairformer(s, z, pz, p, shape_watch_first=True)

        print("\n  [EM-Pairformer Output Shapes Verification]")
        print(f"    Sequence Feature (s): {list(s_out.shape)} (Expected: [1, 20, 64])")
        print(f"    Residue-Residue (z):  {list(z_out.shape)} (Expected: [1, 20, 20, 32])")
        print(f"    Point-Residue (pz):   {list(pz_out.shape)} (Expected: [1, 15, 20, 32])")
        print(f"    Point-Point (p):      {list(p_out.shape)} (Expected: [1, 15, 15, 32])")

        assert s_out.shape == (B, N_res, c_s)
        assert z_out.shape == (B, N_res, N_res, c_z)
        assert pz_out.shape == (B, N_point, N_res, c_pz)
        assert p_out.shape == (B, N_point, N_point, c_p)
        print("  Mathematics and dimensions verified successfully!")

    run_em_pairformer_shape_watcher_demo()

    print("\nLectures 2 & 3 compiled and executed successfully in active Google Colab mode!")
