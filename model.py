import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb
from tqdm import tqdm
from sklearn.model_selection import KFold
from scipy.optimize import linear_sum_assignment
from einops import rearrange


# ==============================================================================
# CONSTANTS & CONFIGURATIONS
# ==============================================================================

RANDOM_SEED = 42
SAVE_DIR = "./pdb_data"

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
BOX_SIZE = 32.0
RADIUS = 1.5

# Mean-Shift Peak Finding Constants
PEAK_THRESHOLD = 0.30
PEAK_BANDWIDTH = 1.0
MAX_PEAKS = 128
PEAK_ITERATIONS = 5
CLASH_LIMIT = 0.6

# Pairformer Hyperparameters
EMBED_DIM_S = 64
EMBED_DIM_Z = 32
NUM_HEADS = 4
NUM_BLOCKS = 3
MAX_REL_POS = 32
CONTACT_THRESHOLD = 8.0

torch.set_default_device('cuda')
device = torch.device("cuda")
print(f"Using device: {device}")

# ==============================================================================
# SECTION 1: RASTERIZATION & DATA PIPELINE
# ==============================================================================

def download_pdb_cif(pdb_id: str) -> str:
    path = rcsb.fetch(pdb_id, "cif", SAVE_DIR)
    if isinstance(path, list):
        return str(path[0])
    return str(path)


def rasterize_structure(coords: torch.Tensor, res_indices: torch.Tensor, sigma: float = 0.8, radius: float = 0.8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized, chunk-free grid rasterization using torch.cdist.
    """
    # coords shape: [N_atoms, 3]
    # res_indices shape: [N_atoms] (long)
    ticks = torch.linspace(0.0, BOX_SIZE, GRID_SIZE)  # [GRID_SIZE]
    grid_x, grid_y, grid_z = torch.meshgrid(ticks, ticks, ticks, indexing='ij')  # each [GRID_SIZE, GRID_SIZE, GRID_SIZE]
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)  # [GRID_SIZE^3, 3]

    dists = torch.cdist(grid, coords)  # [GRID_SIZE^3, N_atoms]
    density = torch.exp(-dists**2 / (2 * sigma**2)).sum(dim=-1).view(GRID_SIZE, GRID_SIZE, GRID_SIZE)  # [GRID_SIZE, GRID_SIZE, GRID_SIZE]
    binary_grid = (dists <= radius).any(dim=-1).float().view(GRID_SIZE, GRID_SIZE, GRID_SIZE)  # [GRID_SIZE, GRID_SIZE, GRID_SIZE]

    min_dists, min_idx = torch.min(dists, dim=-1)  # min_dists: [GRID_SIZE^3], min_idx: [GRID_SIZE^3] (long)
    residue_grid = torch.zeros(grid.shape[0], dtype=torch.long)  # [GRID_SIZE^3] (long)
    valid_mask = min_dists <= radius  # [GRID_SIZE^3] (bool)
    residue_grid[valid_mask] = res_indices[min_idx[valid_mask]]  # [GRID_SIZE^3] (long)

    return density, binary_grid, residue_grid.view(GRID_SIZE, GRID_SIZE, GRID_SIZE)  # returned as density/binary_grid: [GRID_SIZE, GRID_SIZE, GRID_SIZE], residue_grid: [GRID_SIZE, GRID_SIZE, GRID_SIZE] (long)


def crop_and_rasterize_dynamic(structures: list, is_training: bool = False, return_coords: bool = False) -> tuple:
    coords, res_indices = random.choice(structures)  # coords: [N_atoms, 3], res_indices: [N_atoms] (long)
    center = coords[torch.randint(0, len(coords), (1,))].squeeze(0)  # [3]

    half_box = BOX_SIZE / 2.0
    mask = torch.all((coords >= center - half_box) & (coords <= center + half_box), dim=-1)  # [N_atoms] (bool)

    cropped_coords = coords[mask] - center + half_box  # [N_cropped, 3]
    cropped_res = res_indices[mask]  # [N_cropped] (long)

    sigma = random.uniform(0.8, 1.8) if is_training else 1.2
    noise = random.uniform(0.01, 0.08) if is_training else 0.04

    density, binary_grid, residue_grid = rasterize_structure(cropped_coords, cropped_res, sigma=sigma, radius=RADIUS)

    out_density = F.relu(density + torch.randn_like(density) * noise)  # [GRID_SIZE, GRID_SIZE, GRID_SIZE]
    if return_coords:
        return out_density, binary_grid, residue_grid, cropped_coords, cropped_res
    return out_density, binary_grid, residue_grid


def augment_batch_3d_joint(inputs: torch.Tensor, target_atoms: torch.Tensor, target_res: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # inputs shape: [B, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
    # target_atoms shape: [B, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
    # target_res shape: [B, GRID_SIZE, GRID_SIZE, GRID_SIZE] (long)
    for b in range(inputs.shape[0]):
        for dim in (-3, -2, -1):
            if random.random() > 0.5:
                inputs[b] = torch.flip(inputs[b], [dim])  # [1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
                target_atoms[b] = torch.flip(target_atoms[b], [dim])  # [1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
                target_res[b] = torch.flip(target_res[b], [dim])  # [GRID_SIZE, GRID_SIZE, GRID_SIZE] (long)
        for plane in [(-3, -2), (-2, -1), (-3, -1)]:
            k = random.randint(0, 3)
            if k > 0:
                inputs[b] = torch.rot90(inputs[b], k, plane)  # [1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
                target_atoms[b] = torch.rot90(target_atoms[b], k, plane)  # [1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
                target_res[b] = torch.rot90(target_res[b], k, plane)  # [GRID_SIZE, GRID_SIZE, GRID_SIZE] (long)
    return inputs, target_atoms, target_res


class BCEDiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits shape: [B, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
        # target shape: [B, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
        bce = F.binary_cross_entropy_with_logits(logits, target)
        pred = torch.sigmoid(logits)  # [B, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
        p_flat, t_flat = pred.flatten(1), target.flatten(1)  # both [B, GRID_SIZE^3]
        intersection = (p_flat * t_flat).sum(dim=-1)  # [B]
        dice = 1.0 - (2.0 * intersection + 1e-6) / (p_flat.sum(dim=-1) + t_flat.sum(dim=-1) + 1e-6)  # [B]
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
        # x shape: [B, in_c, H, W, D]
        return self.skip(x) + self.net(x)  # returns [B, out_c, H, W, D]


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
        # x shape: [B, channels, H, W, D]
        b, c, h, w, d = x.shape
        h_in = x  # [B, channels, H, W, D]
        x = self.norm(x)  # [B, channels, H, W, D]
        x = rearrange(self.proj_in(x), "b c h w d -> b (h w d) c")  # [B, H*W*D, channels]
        x = x + self.attn(self.norm_attn(x), self.norm_attn(x), self.norm_attn(x))[0]  # [B, H*W*D, channels]
        x = x + self.ff(self.norm_ff(x))  # [B, H*W*D, channels]
        x = rearrange(x, "b (h w d) c -> b c h w d", h=h, w=w, d=d)  # [B, channels, H, W, D]
        return h_in + self.proj_out(x)  # [B, channels, H, W, D]


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
        # x shape: [B, in_channels, H, W, D]
        x1 = self.down1(x)  # [B, f, H, W, D]
        p1 = self.pool1(x1)  # [B, f, H/2, W/2, D/2]
        x2 = self.down2(p1)  # [B, f*2, H/2, W/2, D/2]
        p2 = self.pool2(x2)  # [B, f*2, H/4, W/4, D/4]

        b = self.bottleneck(p2)  # [B, f*4, H/4, W/4, D/4]

        u1 = self.up1(b)  # [B, f*4, H/2, W/2, D/2]
        x3 = self.conv_up1(torch.cat([u1, x2], dim=1))  # concat -> [B, f*6, H/2, W/2, D/2] -> conv -> [B, f*2, H/2, W/2, D/2]

        u2 = self.up2(x3)  # [B, f*2, H, W, D]
        x4 = self.conv_up2(torch.cat([u2, x1], dim=1))  # concat -> [B, f*3, H, W, D] -> conv -> [B, f, H, W, D]

        out = self.out_conv(x4)  # [B, out_channels, H, W, D]
        if return_ds:
            return out, self.ds_conv(x3)  # out: [B, out_channels, H, W, D], ds_conv(x3): [B, out_channels, H/2, W/2, D/2]
        return out  # [B, out_channels, H, W, D]


# ==============================================================================
# SECTION 3: 3D PEAK FINDING (MEAN-SHIFT)
# ==============================================================================

class BatchedMeanShiftPeakFinder3D(nn.Module):
    def forward(self, density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, X, _, _ = density.shape

        max_pooled = F.max_pool3d(density, kernel_size=3, stride=1, padding=1)
        peaks = (density == max_pooled) & (density >= PEAK_THRESHOLD)
        peak_densities = torch.where(peaks.view(B,-1), density.view(B,-1), -1e9)
        topk_vals, topk_indicies = torch.topk(peak_densities, k=MAX_PEAKS, dim=-1)

        ticks = torch.linspace(0.0, BOX_SIZE, X)
        grid = torch.stack(torch.meshgrid(ticks, ticks, ticks, indexing='ij'), dim=-1).view(-1,3)

        seeds_mask = topk_vals > PEAK_THRESHOLD
        seeds = grid[topk_indicies]
        seeds = torch.where(seeds_mask.unsqueeze(-1), seeds, 0.0)

        bandwidth_gaussian = 1.0 / (2 * PEAK_BANDWIDTH ** 2)
        weights_grid = torch.where(density.view(B, -1) > PEAK_THRESHOLD, density.view(B,-1), 0.0)
        for _ in range (PEAK_ITERATIONS):
            sq_dists = torch.cdist(seeds, grid.unsqueeze(0), p=2) ** 2
            weights = torch.exp(-sq_dists * bandwidth_gaussian) * weights_grid.unsqueeze(1)
            denominator = weights.sum(dim=-1, keepdim=True) + 1e-8
            seeds = weights @ grid / denominator

        all_seed_dists = torch.cdist(seeds, seeds, p=2) ** 2
        clash_matrix = torch.triu(all_seed_dists < (CLASH_LIMIT ** 2), diagonal=1)
        keep = seeds_mask.clone()
        for i in range(MAX_PEAKS - 1):
            keep = keep & ~(clash_matrix[:, i, :] & keep[:, i:i+1])

        # Vectorized Packing (Pushes kept peaks contiguously to the front)
        keep_cum = torch.cumsum(keep.long(), dim=-1)
        target_idx = keep_cum - 1

        b_idx_o, seq_idx_o = torch.nonzero(keep, as_tuple=True)
        out_seq_idx = target_idx[b_idx_o, seq_idx_o]

        out_coords = torch.zeros(B, MAX_PEAKS, 3)
        out_coords[b_idx_o, out_seq_idx] = seeds[b_idx_o, seq_idx_o]

        out_mask = torch.zeros(B, MAX_PEAKS, dtype=torch.bool)
        out_mask[b_idx_o, out_seq_idx] = True

        spacing = BOX_SIZE / (X - 1)
        coords_grid = (out_coords / spacing).round().long().clamp(0, X-1)

        flat_coords = coords_grid[..., 0] * (X*X) + coords_grid[..., 1] * X + coords_grid[..., 2]
        lookup_vals = torch.gather(density.view(B, -1), dim=-1, index=flat_coords)
        out_vals = lookup_vals.masked_fill(~out_mask, 0.0)

        return out_coords, out_vals, out_mask

# ==============================================================================
# SECTION 4: THE PAIRFORMER ARCHITECTURE
# ==============================================================================

class RelativePositionEmbedding(nn.Module):
    def __init__(self, max_rel_pos: int = 32, c_z: int = 32):
        super().__init__()
        self.max_rel_pos = max_rel_pos
        self.num_bins = 2 * max_rel_pos + 1
        self.emb = nn.Embedding(self.num_bins, c_z)

    def forward(self, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len)  # [seq_len] (long)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [seq_len, seq_len] (long)
        return self.emb(torch.clamp(diff, -self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos)  # [seq_len, seq_len, c_z]


class SequenceToPairInitializer(nn.Module):
    def __init__(self, c_s: int, c_z: int, max_rel_pos: int = 32):
        super().__init__()
        self.linear_s_q = nn.Linear(c_s, c_z)
        self.linear_s_k = nn.Linear(c_s, c_z)
        self.rel_pos = RelativePositionEmbedding(max_rel_pos, c_z)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s shape: [B, seq_len, c_s]
        s_q = self.linear_s_q(s).unsqueeze(2)  # [B, seq_len, 1, c_z]
        s_k = self.linear_s_k(s).unsqueeze(1)  # [B, 1, seq_len, c_z]
        return s_q + s_k + self.rel_pos(s.shape[1]).unsqueeze(0)  # [B, seq_len, seq_len, c_z]


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
        # s shape: [B, N, c_s]
        # z shape: [B, N, N, c_z]
        B, N, _ = s.shape
        H = self.n_heads

        q = rearrange(self.proj_q(s), "b n (h d) -> b h n d", h=H)  # [B, H, N, d_k]
        k = rearrange(self.proj_k(s), "b n (h d) -> b h n d", h=H)  # [B, H, N, d_k]
        v = rearrange(self.proj_v(s), "b n (h d) -> b h n d", h=H)  # [B, H, N, d_k]

        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.d_k ** 0.5)  # [B, H, N, N]
        bias = rearrange(self.proj_bias(z), "b i j h -> b h i j")  # [B, H, N, N]

        attn_probs = F.softmax(logits + bias, dim=-1)  # [B, H, N, N]
        out = torch.matmul(attn_probs, v)  # [B, H, N, d_k]

        gate = torch.sigmoid(rearrange(self.proj_gate(s), "b n (h d) -> b h n d", h=H))  # [B, H, N, d_k]
        out = rearrange(out * gate, "b h n d -> b n (h d)")  # [B, N, c_s]

        if shape_watch:
            print(f"    [Shape Watcher - AttentionPairBias]\n      s={list(s.shape)}, z={list(z.shape)}, Q/K/V heads={list(q.shape)}")

        return self.proj_out(out)  # [B, N, c_s]


class OuterProductMean(nn.Module):
    def __init__(self, c_s: int, c_z: int, c_hidden: int = 16):
        super().__init__()
        self.ln = nn.LayerNorm(c_s)
        self.lin1 = nn.Linear(c_s, c_hidden)
        self.lin2 = nn.Linear(c_s, c_hidden)
        self.lin_out = nn.Linear(c_hidden ** 2, c_z)

    def forward(self, s: torch.Tensor, shape_watch: bool = False) -> torch.Tensor:
        # s shape: [B, N, c_s]
        s_norm = self.ln(s)  # [B, N, c_s]
        a = self.lin1(s_norm)  # [B, N, c_hidden]
        b = self.lin2(s_norm)  # [B, N, c_hidden]
        outer = rearrange(torch.einsum("b i c, b j d -> b i j c d", a, b), "b i j c d -> b i j (c d)")  # [B, N, N, c_hidden * c_hidden]
        out = self.lin_out(outer)  # [B, N, N, c_z]

        if shape_watch:
            print(f"    [Shape Watcher - OuterProductMean]\n      outer_flat={list(outer.shape)}, z_update={list(out.shape)}")

        return out  # [B, N, N, c_z]


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
        # z shape: [B, N, N, c_z]
        z_norm = self.ln(z)  # [B, N, N, c_z]
        a = torch.sigmoid(self.lin_gate_a(z_norm)) * self.lin_a(z_norm)  # [B, N, N, c_hidden]
        b = torch.sigmoid(self.lin_gate_b(z_norm)) * self.lin_b(z_norm)  # [B, N, N, c_hidden]
        out = torch.einsum("b i k c, b j k c -> b i j c", a, b)  # [B, N, N, c_hidden]
        final_out = self.lin_out(out) * torch.sigmoid(self.lin_gate_out(z_norm))  # [B, N, N, c_z]

        if shape_watch:
            print(f"    [Shape Watcher - TriangleMultiplicativeUpdate]\n      gathered_sum={list(out.shape)}")

        return final_out  # [B, N, N, c_z]


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
        # s shape: [B, N, c_s]
        # z shape: [B, N, N, c_z]
        if shape_watch: print("\n=== STARTING PAIRFORMER BLOCK SHAPE-WATCHING ===")
        s = s + self.attn(self.ln_s1(s), z, shape_watch=shape_watch)  # [B, N, c_s]
        s = s + self.transition_s(s)  # [B, N, c_s]
        z = z + self.opm(s, shape_watch=shape_watch)  # [B, N, N, c_z]
        z = z + self.tri_mul(z, shape_watch=shape_watch)  # [B, N, N, c_z]
        z = z + self.transition_z(z)  # [B, N, N, c_z]
        if shape_watch: print("=== END PAIRFORMER BLOCK SHAPE-WATCHING ===\n")
        return s, z  # ([B, N, c_s], [B, N, N, c_z])


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
        # x shape: [B, N] (long) (residue sequence tokens)
        s = self.embedding(x)  # [B, N, c_s]
        z = self.initializer(s)  # [B, N, N, c_z]
        s, z = self.pairformer(s, z, shape_watch_first=shape_watch)  # ([B, N, c_s], [B, N, N, c_z])
        return self.contact_head(z).squeeze(-1)  # [B, N, N]


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
# SECTION 4B: THE EM-PAIRFORMER ARCHITECTURE
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
        p_norm = self.ln_p(p)  # [B, N_point, N_point, c_p]
        z_norm = self.ln_z(z)  # [B, N_res, N_res, c_z]

        x_p = torch.sigmoid(self.lin_p_gate(p_norm)) * self.lin_p_proj(p_norm)  # [B, N_point, N_point, c_hidden]
        x_z = torch.sigmoid(self.lin_z_gate(z_norm)) * self.lin_z_proj(z_norm)  # [B, N_res, N_res, c_hidden]

        if self.outgoing:
            # sum over intermediate connections to find outgoing paths (dim -2)
            z_p = x_p.sum(dim=-2)  # [B, N_point, c_hidden]
            z_z = x_z.sum(dim=-2)  # [B, N_res, c_hidden]
        else:
            # incoming paths (dim -3)
            z_p = x_p.sum(dim=-3)  # [B, N_point, c_hidden]
            z_z = x_z.sum(dim=-3)  # [B, N_res, c_hidden]

        out = torch.einsum("b p h, b r h -> b p r h", z_p, z_z)  # [B, N_point, N_res, c_hidden]
        return self.lin_out(self.ln_out(out))  # [B, N_point, N_res, c_pz]


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
        # pz shape: [B, N_point, N_res, c_pz]
        # bias_tensor shape: [B, N_res, N_res, c_bias] or [B, N_point, N_point, c_bias]
        H = self.n_heads
        pz_norm = self.ln_pz(pz)  # [B, N_point, N_res, c_pz]

        pat = "b p r (h d) -> b p h r d" if self.along_dim == -2 else "b p r (h d) -> b r h p d"
        q, k, v = [rearrange(proj(pz_norm), pat, h=H) for proj in (self.proj_q, self.proj_k, self.proj_v)]  # each [B, N_point, H, N_res, d] or [B, N_res, H, N_point, d]

        bias = rearrange(self.proj_bias(self.ln_bias(bias_tensor)), "b i j h -> b h i j").unsqueeze(1)  # [B, 1, H, N_dim, N_dim]
        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.d_k ** 0.5)  # [B, N_point, H, N_res, N_res] or [B, N_res, H, N_point, N_point]
        attn_probs = F.softmax(logits + bias, dim=-1)  # [B, N_point, H, N_res, N_res] or [B, N_res, H, N_point, N_point]

        gate = torch.sigmoid(rearrange(self.proj_gate(pz_norm), pat, h=H))  # [B, N_point, H, N_res, d] or [B, N_res, H, N_point, d]
        out_pat = "b p h r d -> b p r (h d)" if self.along_dim == -2 else "b r h p d -> b p r (h d)"
        out = rearrange(torch.matmul(attn_probs, v) * gate, out_pat)  # [B, N_point, N_res, c_hidden]
        return self.proj_out(out)  # [B, N_point, N_res, c_pz]


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
        # m shape: [B, S, N, c_m]
        m_norm = self.ln(m)  # [B, S, N, c_m]
        a = self.lin1(m_norm)  # [B, S, N, c_hidden]
        b = self.lin2(m_norm)  # [B, S, N, c_hidden]
        outer = torch.einsum("b s i c, b s j d -> b i j c d", a, b).flatten(start_dim=-2)  # [B, N, N, c_hidden * c_hidden]
        out = self.lin_out(outer) / (m.shape[1] + 1e-8)  # [B, N, N, c_z]
        return out  # [B, N, N, c_z]


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
        # s shape: [B, N_res, c_s] or None
        # z shape: [B, N_res, N_res, c_z]
        # pz shape: [B, N_point, N_res, c_pz]
        # p shape: [B, N_point, N_point, c_p]
        if shape_watch:
            print(f"    [Shape Watcher - EMPairformerBlock Input]\n      s={list(s.shape) if s is not None else None}, z={list(z.shape)}, pz={list(pz.shape)}, p={list(p.shape)}")

        z = z + self.tri_mul_out(z)  # [B, N_res, N_res, c_z]
        z = z + self.tri_mul_in(z)  # [B, N_res, N_res, c_z]

        pz = pz + self.inter_outgoing(z, p)  # [B, N_point, N_res, c_pz]
        pz = pz + self.inter_incoming(z, p)  # [B, N_point, N_res, c_pz]
        pz = pz + self.residue_row_attn(pz, z)  # [B, N_point, N_res, c_pz]
        pz = pz + self.point_column_attn(pz, p)  # [B, N_point, N_res, c_pz]
        pz = pz + self.pz_transition(pz)  # [B, N_point, N_res, c_pz]

        # update z from pz (averaging over point axis)
        z = z + self.outer_row_opm(pz)  # [B, N_res, N_res, c_z]
        # update p from pz (averaging over residue axis)
        p = p + self.outer_col_opm(pz.transpose(1, 2))  # [B, N_point, N_point, c_p]
        z = z + self.transition_z(z)  # [B, N_res, N_res, c_z]

        if self.c_s > 0 and s is not None:
            s = s + self.attn(s, z, shape_watch=shape_watch)  # [B, N_res, c_s]
            s = s + self.transition_s(s)  # [B, N_res, c_s]

        return s, z, pz, p  # shapes: ([B, N_res, c_s], [B, N_res, N_res, c_z], [B, N_point, N_res, c_pz], [B, N_point, N_point, c_p])


class EMPairformerStack(nn.Module):
    def __init__(self, c_s: int, c_z: int, c_pz: int, c_p: int, n_blocks: int = 3, n_heads: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([EMPairformerBlock(c_s, c_z, c_pz, c_p, n_heads) for _ in range(n_blocks)])

    def forward(self, s: torch.Tensor | None, z: torch.Tensor, pz: torch.Tensor, p: torch.Tensor, shape_watch_first: bool = False) -> tuple:
        for i, block in enumerate(self.blocks):
            s, z, pz, p = block(s, z, pz, p, shape_watch=(shape_watch_first and i == 0))
        return s, z, pz, p

# ==============================================================================
# SECTION 5: K-FOLD AND GOOGLE COLAB TRAINING PIPELINE
# ==============================================================================

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("===========================================================================")
    print(" PREPARING MOLECULAR DATABASE ")
    print("===========================================================================")

    os.makedirs(SAVE_DIR, exist_ok=True)
    all_structures = {}
    for pid in tqdm(PDB_IDS, desc="Caching mmCIF files"):
        filepath = download_pdb_cif(pid)
        try:
            atoms = pdbx.get_structure(pdbx.CIFFile.read(filepath), model=1)
            # 1. 3D Volumetric representations (All atoms)
            valid_atoms = atoms[np.isin(atoms.res_name, list(ALL_RESIDUES))]
            v_coords = torch.tensor(valid_atoms.coord, dtype=torch.float32)  # [N_atoms, 3]
            v_res_idx = torch.tensor([RESIDUE_MAP[name] for name in valid_atoms.res_name], dtype=torch.long)  # [N_atoms] (long)

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
            s_seq = torch.tensor([RESIDUE_MAP.get(name, 0) for name in rep_atoms.res_name], dtype=torch.long)  # [seq_len] (long)
            s_coords = torch.tensor(rep_atoms.coord, dtype=torch.float32)  # [seq_len, 3]

            all_structures[pid] = {
                "v_coords": v_coords, "v_res_idx": v_res_idx,
                "s_seq": s_seq, "s_coords": s_coords
            }
        except Exception as e:
            print(f"Failed to load PDB {pid}: {e}")

    # --------------------------------------------------------------------------
    # DEMO 1: 5-FOLD 3D VOLUMETRIC U-NET CROSS-VALIDATION
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" PIPELINE 1: TRAINING 5-FOLD VOLUMETRIC U-NET DEMO ")
    print("="*70)

    K_FOLDS = 5
    NUM_EPOCHS = 1 if device.type == "cpu" else 100
    steps_per_epoch = 1 if device.type == "cpu" else 25
    MATCHING_RADIUS = 1.5
    spacing = BOX_SIZE / (GRID_SIZE - 1)

    # Initialize 5 folds on the 8 PDBs using scikit-learn
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    pdb_folds = [list(np.array(PDB_IDS)[test_idx]) for _, test_idx in kf.split(PDB_IDS)]
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

        test_pids = pdb_folds[fold_idx]
        val_pids = pdb_folds[(fold_idx + 1) % K_FOLDS]
        train_pids = [pid for i, fold in enumerate(pdb_folds) if i not in (fold_idx, (fold_idx + 1) % K_FOLDS) for pid in fold]
        print(f"Train structures ({len(train_pids)}): {[p.upper() for p in train_pids]}")
        print(f"Validation structures ({len(val_pids)}): {[p.upper() for p in val_pids]}")
        print(f"Test structures ({len(test_pids)}): {[p.upper() for p in test_pids]}")

        # Construct splits
        v_train = [(all_structures[pid]["v_coords"], all_structures[pid]["v_res_idx"]) for pid in train_pids]
        v_val = [(all_structures[pid]["v_coords"], all_structures[pid]["v_res_idx"]) for pid in val_pids]
        v_test = [(all_structures[pid]["v_coords"], all_structures[pid]["v_res_idx"]) for pid in test_pids]

        # Initialize models
        unet_atom = torch.compile(UNet3D(1, 1, init_features=16))
        unet_res = torch.compile(UNet3D(1, len(RESIDUE_MAP) + 1, init_features=16))
        opt_unet = torch.optim.Adam(list(unet_atom.parameters()) + list(unet_res.parameters()), lr=0.001)
        criterion_atom = BCEDiceLoss()
        criterion_res = nn.CrossEntropyLoss(ignore_index=0)

        def compute_unet_loss(pred_atom, ds_atom, target_atoms, pred_res, ds_res, target_res):
            # pred_atom shape: [B, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
            # ds_atom shape: [B, 1, GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE/2]
            # target_atoms shape: [B, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
            # pred_res shape: [B, C_res, GRID_SIZE, GRID_SIZE, GRID_SIZE]
            # ds_res shape: [B, C_res, GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE/2]
            # target_res shape: [B, GRID_SIZE, GRID_SIZE, GRID_SIZE] (long)
            loss_atom_main = criterion_atom(pred_atom, target_atoms)
            loss_res_main = criterion_res(pred_res, target_res)
            target_atoms_ds = F.max_pool3d(target_atoms, kernel_size=2, stride=2)  # [B, 1, GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE/2]
            target_res_ds = F.max_pool3d(target_res.float().unsqueeze(1), kernel_size=2, stride=2).squeeze(1).long()  # [B, GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE/2] (long)
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
                inputs = torch.stack([s[0] for s in samples]).unsqueeze(1)  # [4, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
                target_atoms = torch.stack([s[1] for s in samples]).unsqueeze(1)  # [4, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
                target_res = torch.stack([s[2] for s in samples]).long()  # [4, GRID_SIZE, GRID_SIZE, GRID_SIZE] (long)

                inputs, target_atoms, target_res = augment_batch_3d_joint(inputs, target_atoms, target_res)  # shapes same as above

                opt_unet.zero_grad()
                pred_atom, ds_atom = unet_atom(inputs, return_ds=True)  # pred_atom: [4, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE], ds_atom: [4, 1, GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE/2]
                pred_res, ds_res = unet_res(inputs, return_ds=True)  # pred_res: [4, C_res, GRID_SIZE, GRID_SIZE, GRID_SIZE], ds_res: [4, C_res, GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE/2]

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
                        val_inputs = torch.stack([s[0] for s in val_samples]).unsqueeze(1)  # [4, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
                        val_target_atoms = torch.stack([s[1] for s in val_samples]).unsqueeze(1)  # [4, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
                        val_target_res = torch.stack([s[2] for s in val_samples]).long()  # [4, GRID_SIZE, GRID_SIZE, GRID_SIZE] (long)

                        val_pred_atom, val_ds_atom = unet_atom(val_inputs, return_ds=True)  # shapes same as above
                        val_pred_res, val_ds_res = unet_res(val_inputs, return_ds=True)  # shapes same as above
                        val_loss += compute_unet_loss(val_pred_atom, val_ds_atom, val_target_atoms, val_pred_res, val_ds_res, val_target_res).item()
                    val_loss /= val_steps

                print(f"Fold {fold_idx + 1} | Epoch {epoch:03d}/100 | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_atom_state = {k: v.clone() for k, v in unet_atom.state_dict().items()}
                    best_res_state = {k: v.clone() for k, v in unet_res.state_dict().items()}

        # Restore best model for testing
        if best_atom_state is not None:
            unet_atom.load_state_dict({k: v.to(device) for k, v in best_atom_state.items()})
            unet_res.load_state_dict({k: v.to(device) for k, v in best_res_state.items()})

        # Test Evaluation for this fold
        print(f"\nEvaluating Fold {fold_idx + 1} on unseen test structures...")
        unet_atom.eval(); unet_res.eval()
        # Test Evaluation for this fold
        print(f"\nEvaluating Fold {fold_idx + 1} on unseen test structures...")
        unet_atom.eval(); unet_res.eval()
        peak_finder = BatchedMeanShiftPeakFinder3D()

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
                    )  # test_input: [GRID_SIZE, GRID_SIZE, GRID_SIZE], gt_coords: [N_cropped, 3], gt_res_indices: [N_cropped] (long)
                    test_in_batch = test_input.unsqueeze(0).unsqueeze(0)  # [1, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]

                    pred_density = F.relu(unet_atom(test_in_batch))  # [1, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE]
                    pred_coords, _, pred_mask = peak_finder(pred_density)  # pred_coords: [1, M, 3], pred_mask: [1, M] (bool)
                    pred_res_logits = unet_res(test_in_batch)  # [1, C_res, GRID_SIZE, GRID_SIZE, GRID_SIZE]

                    pred_coords_gpu = pred_coords[0]  # [M, 3]
                    num_pred_peaks = pred_mask[0].sum().item()  # scaler

                    pid_resolved_peaks += num_pred_peaks
                    pid_gt_atoms += len(gt_coords)

                    if num_pred_peaks > 0 and len(gt_coords) > 0:
                        dists = torch.cdist(gt_coords, pred_coords_gpu[:num_pred_peaks])  # [N_cropped, num_pred_peaks]
                        row_ind, col_ind = linear_sum_assignment(dists.cpu().numpy())
                        for r, c in zip(row_ind, col_ind):
                            dist = dists[r, c].item()
                            if dist <= MATCHING_RADIUS:
                                pid_matched_atoms += 1
                                p_coord = pred_coords_gpu[c]  # [3]
                                grid_idx = torch.clamp(torch.round(p_coord / spacing).long(), 0, GRID_SIZE - 1)  # [3] (long)
                                logits = pred_res_logits[0, :, grid_idx[0], grid_idx[1], grid_idx[2]]  # [C_res]
                                if torch.argmax(logits[1:]).item() + 1 == gt_res_indices[r].item():
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
    # DEMO 2: PAIRFORMER SEQUENCE-TO-PAIR OPTIMIZATION
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" PIPELINE 2: TRAINING PAIRFORMER CONTACT MAP OPTIMIZATION ")
    print("="*70)

    pairformer = torch.compile(PairformerContactPredictor(
        vocab_size=len(RESIDUE_MAP), c_s=EMBED_DIM_S, c_z=EMBED_DIM_Z, n_blocks=NUM_BLOCKS, n_heads=NUM_HEADS
    ))
    opt_pf = torch.optim.Adam(pairformer.parameters(), lr=0.002)
    criterion_pf = nn.BCEWithLogitsLoss()

    # Precompute static target contacts to optimize speed and reduce lines
    pf_dataset = []
    for pid, s in all_structures.items():
        tokens = s["s_seq"].unsqueeze(0)  # [1, seq_len] (long)
        coords = s["s_coords"]  # [seq_len, 3]
        target = (torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0) < CONTACT_THRESHOLD).float().unsqueeze(0)  # [1, seq_len, seq_len]
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
            loss = criterion_pf(pairformer(tokens), target)  # scaler
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
        vis_tokens = vis_s["s_seq"].unsqueeze(0)  # [1, seq_len] (long)
        vis_pred = torch.sigmoid(pairformer(vis_tokens, shape_watch=True)).squeeze(0).cpu()  # [seq_len, seq_len]
        vis_gt = (torch.cdist(vis_s["s_coords"].unsqueeze(0), vis_s["s_coords"].unsqueeze(0)).squeeze(0) < CONTACT_THRESHOLD).float()  # [seq_len, seq_len]
        print(f"\nCompleted run. Contact Map ASCII Visual check for {vis_key.upper()}:")
        print_ascii_contact_map(vis_gt, vis_pred, threshold=0.5)
    print("\n" + "="*70)
    print(" PIPELINE 3: MATHEMATICAL DYNAMIC SHAPE WATCHING OF EM-PAIRFORMER ")
    print("="*70)

    # Shape watcher unit tests for the EM-Pairformer (The Karpathy Touch!)
    def run_em_pairformer_shape_watcher_demo():
        print("\n[Shape Watcher] Initializing EM-Pairformer Demo...")
        B, N_res, N_point = 1, 20, 15
        c_s, c_z, c_pz, c_p = 64, 32, 32, 32

        # 1. Mock inputs matching physical coordinates and densities
        vocab_size = len(RESIDUE_MAP)
        mock_tokens = torch.randint(1, vocab_size, (B, N_res))  # [B, N_res] (long)
        mock_p_coords = torch.rand(B, N_point, 3) * BOX_SIZE  # [B, N_point, 3]
        mock_p_densities = torch.rand(B, N_point)  # [B, N_point]
        mock_res_coords = torch.rand(B, N_res, 3) * BOX_SIZE  # [B, N_res, 3]

        # 2. Build physical representation matrices using RBF
        def mock_rbf_expansion(dists: torch.Tensor, num_centers: int = 32) -> torch.Tensor:
            # dists: [B, N_dim1, N_dim2]
            centers = torch.linspace(0.0, BOX_SIZE, num_centers)  # [num_centers]
            widths = BOX_SIZE / num_centers
            return torch.exp(-((dists.unsqueeze(-1) - centers) / widths) ** 2)  # [B, N_dim1, N_dim2, num_centers]

        print("  Generating geometric representation matrices...")
        dist_pp = torch.cdist(mock_p_coords, mock_p_coords)  # [B, N_point, N_point]
        dist_pr = torch.cdist(mock_p_coords, mock_res_coords)  # [B, N_point, N_res]

        p_geom = mock_rbf_expansion(dist_pp, num_centers=c_p)  # [B, N_point, N_point, c_p]
        pz_geom = mock_rbf_expansion(dist_pr, num_centers=c_pz)  # [B, N_point, N_res, c_pz]

        # Inject density features into point representations
        proj_density = nn.Linear(1, c_p)  # maps [B, N_point, 1] -> [B, N_point, c_p]
        p = p_geom + proj_density(mock_p_densities.unsqueeze(-1)).unsqueeze(1)  # [B, N_point, N_point, c_p]
        pz = pz_geom  # [B, N_point, N_res, c_pz]

        # Embed sequence and initialize residue pairs
        embedding = nn.Embedding(vocab_size + 1, c_s, padding_idx=0)
        initializer = SequenceToPairInitializer(c_s, c_z, MAX_REL_POS)

        s = embedding(mock_tokens)  # [B, N_res, c_s]
        z = initializer(s)  # [B, N_res, N_res, c_z]

        # 3. Instantiate EMPairformerStack
        print("  Instantiating EMPairformerStack...")
        em_pairformer = torch.compile(EMPairformerStack(c_s=c_s, c_z=c_z, c_pz=c_pz, c_p=c_p, n_blocks=2, n_heads=4))
        em_pairformer.eval()

        # 4. Forward pass under dynamic Shape Watcher
        print("  Running EM-Pairformer forward pass:")
        with torch.no_grad():
            s_out, z_out, pz_out, p_out = em_pairformer(s, z, pz, p, shape_watch_first=True)  # shapes: ([B, N_res, c_s], [B, N_res, N_res, c_z], [B, N_point, N_res, c_pz], [B, N_point, N_point, c_p])

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
