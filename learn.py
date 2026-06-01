import torch
import torch.nn as nn
import torch.nn.functional as F

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Convo(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, 3, padding=1)
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
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

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
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


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super().__init__()
        f = features
        # Encoder (Downsampling Path)
        self.down1 = Convo(in_channels, f)
        self.pool1 = nn.Conv3d(f, f, kernel_size=3, stride=2, padding=1)

        self.down2 = Convo(f, f * 2)
        self.pool2 = nn.Conv3d(f * 2, f * 2, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            Convo(f * 2, f * 4),
            SpatialTransformer(f * 4, n_heads=2)
        )

        # Decoder (Upsampling Path)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(f * 4, f * 4, kernel_size=3, padding=1)
        )
        self.conv_up1 = Convo(f * 6, f * 2)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(f * 2, f * 2, kernel_size=3, padding=1)
        )
        self.conv_up2 = Convo(f * 3, f)

        # Prediction Heads
        self.out_conv = nn.Conv3d(f, out_channels, 1)
        self.ds_conv = nn.Conv3d(f * 2, out_channels, 1)

    def forward(self, x: torch.Tensor, return_ds: bool = False) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # Downsample
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        x2 = self.down2(p1)
        p2 = self.pool2(x2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Upsample & Skip Connections
        u1 = self.up1(b)
        x3 = self.conv_up1(torch.cat([u1, x2], dim=1))

        u2 = self.up2(x3)
        x4 = self.conv_up2(torch.cat([u2, x1], dim=1))

        # Output
        out = self.out_conv(x4)
        if return_ds:
            return out, self.ds_conv(x3)
        return out

class PeakFinder(nn.Module):
    def forward(self, density):
        B, _, X, _, _ = density.shape
        M = MAX_PEAKS
        spacing = BOX_SIZE / (X-1)

        ticks = torch.linspace(0.0, BOX_SIZE, X)
        grid = torch.stack(torch.meshgrid(ticks, ticks, ticks, indexing='ij'), dim=-1).view(-1,3)

        peaks = (density == F.max_pool3d(density, kernel_size=3, stride=1, padding=1)) & (density > PEAK_THRESHOLD)
        max_active = ((density > PEAK_THRESHOLD).view(B, -1)).sum(dim=-1).max().item()

        peaks_mask = peaks.view(B, -1)
        cum_sum_peaks = torch.cumsum(peaks_mask, dim=-1)
        peak_positions = (cum_sum_peaks * peaks_mask) -1
        S_max = peaks_mask & (peak_positions < M).sum(dim=-1).max().item()

        out_coords = torch.zeros (B, M, 3)
        out_vals = torch.zeros(B, M)
        out_mask = torch.zeros(B, M, dtype=torch.bool)

        if max_active == 0 or S_max == 0:
            return out_coords, out_vals, out_mask








