import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# SECTION 1: SPATIAL DENSITY GENERATION (With [X, Y, Z] Grid Alignment)
# ==============================================================================

from lec1 import cif_to_density


# Generate input (all atoms) and target (Carbon atoms only)
input_density: torch.Tensor = cif_to_density("data.cif")                      # Includes N, CA, C, O
target_density: torch.Tensor = cif_to_density("data.cif", element_filter="C")  # Includes only CA and C


# ==============================================================================
# SECTION 2: THE 3D U-NET ARCHITECTURE
# ==============================================================================
"""
The 3D U-Net downsamples spatial grids using pooling, processes latent features
at a lower resolution, and then upsamples to reconstruct the dense volume.

The skip connections concatenate raw spatial feature maps from the encoder path
directly into the decoder path, preserving sub-angstrom boundaries.
"""

class DoubleConv3D(nn.Module):
    """(Conv3D -> BatchNorm3D -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
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
    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        # --- Encoder (Downsampling Path) ---
        self.down1: DoubleConv3D = DoubleConv3D(in_channels, 8)
        self.pool1: nn.MaxPool3d = nn.MaxPool3d(kernel_size=2, stride=2) # 20^3 -> 10^3

        self.down2: DoubleConv3D = DoubleConv3D(8, 16)
        self.pool2: nn.MaxPool3d = nn.MaxPool3d(kernel_size=2, stride=2) # 10^3 -> 5^3

        # --- Bottleneck ---
        self.bottleneck: DoubleConv3D = DoubleConv3D(16, 32)

        # --- Decoder (Upsampling Path) ---
        self.up1: nn.ConvTranspose3d = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2) # 5^3 -> 10^3
        self.conv_up1: DoubleConv3D = DoubleConv3D(32, 16) # Input channels: 16 (upsampled) + 16 (skip)

        self.up2: nn.ConvTranspose3d = nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2)  # 10^3 -> 20^3
        self.conv_up2: DoubleConv3D = DoubleConv3D(16, 8) # Input channels: 8 (upsampled) + 8 (skip)

        # Output Projection
        self.out_conv: nn.Conv3d = nn.Conv3d(8, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Channel, X_dim, Y_dim, Z_dim) -> (B, 1, 20, 20, 20)

        # Encoder
        x1: torch.Tensor = self.down1(x)              # (B, 8, 20, 20, 20)
        p1: torch.Tensor = self.pool1(x1)              # (B, 8, 10, 10, 10)

        x2: torch.Tensor = self.down2(p1)              # (B, 16, 10, 10, 10)
        p2: torch.Tensor = self.pool2(x2)              # (B, 16, 5, 5, 5)

        # Bottleneck
        b: torch.Tensor = self.bottleneck(p2)          # (B, 32, 5, 5, 5)

        # Decoder
        u1: torch.Tensor = self.up1(b)                 # (B, 16, 10, 10, 10)
        c1: torch.Tensor = torch.cat([u1, x2], dim=1)  # (B, 32, 10, 10, 10) - Skip Connection
        x3: torch.Tensor = self.conv_up1(c1)           # (B, 16, 10, 10, 10)

        u2: torch.Tensor = self.up2(x3)                # (B, 8, 20, 20, 20)
        c2: torch.Tensor = torch.cat([u2, x1], dim=1)  # (B, 16, 20, 20, 20) - Skip Connection
        x4: torch.Tensor = self.conv_up2(c2)           # (B, 8, 20, 20, 20)

        return self.out_conv(x4)         # (B, 1, 20, 20, 20)


# ==============================================================================
# SECTION 3: 3D NON-MAXIMUM SUPPRESSION (PEAK-FINDER)
# ==============================================================================
"""
A trained network outputs a smooth density heatmap. We extract point coordinates
by finding local maxima in a localized 3D search window.
"""

def find_peaks_3d(density: torch.Tensor, threshold: float = 0.15, kernel_size: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Finds local maxima in a 3D density map.

    Args:
        density (Tensor): Continuous density map of shape (B, C, X_dim, Y_dim, Z_dim)
        threshold (float): Minimum value to classify as a peak
        kernel_size (int): Dimensions of localized search space
    """
    padding: int = kernel_size // 2

    # Keep only local maxima in the 3x3x3 neighborhood
    max_pooled: torch.Tensor = F.max_pool3d(density, kernel_size=kernel_size, stride=1, padding=padding)

    # Identify positions matching original values and exceeding the intensity cutoff
    is_peak: torch.Tensor = (density == max_pooled) & (density > threshold)

    # Extract coordinate indices: (num_peaks, 5) -> [Batch, Channel, X_idx, Y_idx, Z_idx]
    peak_indices: torch.Tensor = torch.nonzero(is_peak)
    spatial_indices: torch.Tensor = peak_indices[:, 2:] # Strip out Batch and Channel -> [X_idx, Y_idx, Z_idx]
    values: torch.Tensor = density[is_peak]

    return spatial_indices, values


# ==============================================================================
# SECTION 4: TRAINING & COORDINATE DECODING
# ==============================================================================
if __name__ == "__main__":
    from lec2 import UNet3D, BatchedPeakFinder3D
    
    # Setup training configurations
    torch.manual_seed(42)
    model: UNet3D = UNet3D(in_channels=1, out_channels=1, init_features=8)
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion: nn.Module = nn.MSELoss()

    # Create 5D batch tensors: (Batch, Channel, X_dim, Y_dim, Z_dim)
    input_batch: torch.Tensor = input_density.unsqueeze(0).unsqueeze(0)   # (1, 1, 20, 20, 20)
    target_batch: torch.Tensor = target_density.unsqueeze(0).unsqueeze(0) # (1, 1, 20, 20, 20)

    # Initialize our robust, batched peak finder (support point extractor)
    peak_finder = BatchedPeakFinder3D(threshold=0.15, max_peaks=16, box_size=10.0)

    print("="*60)
    print(" BASELINE GROUND TRUTH (USING BATCHED PEAK FINDER) ")
    print("="*60)
    print("Extracting physical peaks directly from Target Density Map:")
    
    with torch.no_grad():
        gt_coords, gt_vals, gt_mask = peak_finder(target_batch)
        
    # Batch index 0 contains our target macromolecule peaks
    num_gt_peaks = gt_mask[0].sum().item()
    for idx in range(num_gt_peaks):
        coord = gt_coords[0, idx]
        val = gt_vals[0, idx]
        print(f"  Target Carbon Peak {idx+1}: X={coord[0]:.3f}, Y={coord[1]:.3f}, Z={coord[2]:.3f} | Confidence={val:.4f}")

    # Training
    print("\n" + "="*60)
    print(" RUNNING 3D U-NET OVERFITTING ")
    print("="*60)

    epochs: int = 150
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        output: torch.Tensor = model(input_batch)
        loss: torch.Tensor = criterion(output, target_batch)

        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Training Loss: {loss.item():.6f}")

    # Evaluate model predictions
    model.eval()
    with torch.no_grad():
        predicted_batch: torch.Tensor = model(input_batch)
        predicted_batch = F.relu(predicted_batch) # Clamp negative outputs to 0

    # Extract peaks from the model's prediction output using the batched peak-finder
    with torch.no_grad():
        pred_coords, pred_vals, pred_mask = peak_finder(predicted_batch)

    print("\n" + "="*60)
    print(" RESOLVED INFERENCE ANALYSIS (USING BATCHED PEAK FINDER) ")
    print("="*60)
    
    num_pred_peaks = pred_mask[0].sum().item()
    if num_pred_peaks == 0:
        print("No peaks resolved. Verify model convergence or adjust threshold.")
    else:
        for idx in range(num_pred_peaks):
            coord = pred_coords[0, idx]
            val = pred_vals[0, idx]
            print(f"  Model Recovered Peak {idx+1}: X={coord[0]:.3f}, Y={coord[1]:.3f}, Z={coord[2]:.3f} | Confidence={val:.4f}")

