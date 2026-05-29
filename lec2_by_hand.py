import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import biotite.structure.io.pdbx as pdbx

# =====================================================================
# 1. DATA LOADING & 3D VECTORIZED RASTERIZATION
# =====================================================================
# We load real protein structures from the cached PDB folder.
# - We filter for Carbon-alpha (CA) atoms (exactly one coordinate per residue).
# - We center the protein and rasterize it onto a 3D grid using standard Gaussian blurs.
# - Clean Target: sharp Gaussian density peaks at atom coordinates (sigma = 1.0).
# - Noisy Input: wide, blurred peaks + random noise (simulating low-res cryo-EM maps, sigma = 2.0).

def load_coords_biotite(filepath):
    # Read the mmCIF file using biotite
    cif_file = pdbx.CIFFile.read(filepath)
    atoms = pdbx.get_structure(cif_file, model=1)
    
    # Filter for C-alpha atoms in protein residues (1 point per residue)
    valid_atoms = atoms[(atoms.res_name != "HOH") & (atoms.atom_name == "CA")]
    
    # Convert coordinates to PyTorch tensor
    return torch.tensor(valid_atoms.coord, dtype=torch.float32)


def rasterize_to_3d_grid(coords, grid_size=32, box_size=32.0, sigma=1.5, noise_std=0.0):
    device = coords.device
    
    # 1. Center the coordinates in our 3D box of size [0, box_size]^3
    center = coords.mean(dim=0)
    centered_coords = coords - center + (box_size / 2.0)
    
    # 2. Set up spatial grid ticks and meshgrid
    ticks = torch.linspace(0.0, box_size, grid_size, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(ticks, ticks, ticks, indexing='ij')
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1) # Shape: [32, 32, 32, 3]
    g_flat = grid.view(-1, 3) # Flattened grid voxels: Shape [32768, 3]
    
    # 3. Vectorized distance computation between all grid voxels and all atoms
    # Formula: (A - B)^2 = A^2 + B^2 - 2AB
    g2 = torch.sum(g_flat ** 2, dim=-1, keepdim=True) # [32768, 1]
    c2 = torch.sum(centered_coords ** 2, dim=-1, keepdim=True).t() # [1, N]
    
    sq_dists = g2 + c2 - 2.0 * torch.matmul(g_flat, centered_coords.t())
    sq_dists = torch.clamp(sq_dists, min=0.0)
    
    # 4. Generate the Gaussian density grid
    density_flat = torch.sum(torch.exp(-sq_dists / (2 * (sigma ** 2))), dim=-1)
    density_grid = density_flat.view(grid_size, grid_size, grid_size)
    
    # Cap values at 1.0
    density_grid = torch.clamp(density_grid, max=1.0)
    
    # Add random noise
    if noise_std > 0.0:
        density_grid = density_grid + torch.randn_like(density_grid) * noise_std
        density_grid = torch.clamp(density_grid, min=0.0)
        
    return density_grid, centered_coords


# =====================================================================
# 2. DEFINING THE 3D NEURAL NETWORK (The Voxel Finder)
# =====================================================================
# Any PyTorch neural network inherits from nn.Module.
# We define layers in __init__() and the data flow in forward().

class Tiny3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv3d expects input shapes: [Batch, Channels, Depth, Height, Width]
        self.net = nn.Sequential(
            # Layer 1: Conv from 1 channel to 16 channels, using a 3x3x3 sliding kernel
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Layer 2: Conv from 16 channels to 16 channels
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Layer 3: Conv from 16 channels back to 1 channel (predicted probability map)
            nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # We apply a Sigmoid at the end to force predicted values between 0.0 and 1.0
        return torch.sigmoid(self.net(x))


# =====================================================================
# 3. HYBRID LOSS FUNCTION & TRAINING LOOP
# =====================================================================
# Because a protein structure is highly sparse (most of the 3D grid is empty space),
# a standard loss function like MSE alone will fail because the model learns to
# output all zeros.
# We use a Hybrid Loss: MSE + Soft Dice Loss.
# Soft Dice calculates overlap (intersection over union), completely ignoring
# the background size and forcing the model to perfectly resolve the atom peaks!

def hybrid_loss(pred, target):
    # 1. Mean Squared Error Loss
    mse = F.mse_loss(pred, target)
    
    # 2. Soft Dice Loss
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
    
    # Combine them (both are in range [0, 1])
    return mse + dice


def train_model(model, train_coords, num_epochs=120, lr=0.002):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("Training the 3D Coordinate-Learning Network...")
    for epoch in range(1, num_epochs + 1):
        # Generate the clean target and noisy input grids independently
        clean_tgt, _ = rasterize_to_3d_grid(train_coords, sigma=1.0, noise_std=0.0)
        noisy_in, _ = rasterize_to_3d_grid(train_coords, sigma=2.0, noise_std=0.04)
        
        # Add Batch and Channel dimensions: Shape [1, 1, 32, 32, 32]
        x = noisy_in.unsqueeze(0).unsqueeze(0)
        y = clean_tgt.unsqueeze(0).unsqueeze(0)
        
        # 1. Forward Pass
        pred = model(x)
        
        # 2. Loss computation
        loss = hybrid_loss(pred, y)
        
        # 3. Backward Pass & Parameter Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 15 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{num_epochs} | Hybrid Loss: {loss.item():.6f}")


# =====================================================================
# 4. 3D PEAK FINDING (Non-Maximum Suppression)
# =====================================================================
# We resolve coordinates by finding local maxima on our predicted grid.
# We do this using standard 3D Max Pooling:
# If a voxel's value is equal to the maximum in its 3x3x3 neighborhood, it's a peak!

def find_peaks_3d(predicted_grid, box_size=32.0, grid_size=32, threshold=0.35):
    # Max pool keeps the maximum value in every local 3x3x3 window
    max_pooled = F.max_pool3d(predicted_grid, kernel_size=3, stride=1, padding=1)
    
    # A voxel is a peak if it equals the local maximum AND exceeds our threshold
    is_peak = (predicted_grid == max_pooled) & (predicted_grid > threshold)
    
    # Extract indices:nonzero() returns shape [N, 5]: [Batch, Channel, Z, Y, X]
    peak_indices = torch.nonzero(is_peak)
    
    # Extract coordinate indices [Z, Y, X]
    grid_coords = peak_indices[:, 2:] # Shape: [N, 3]
    
    # Convert voxel indices back to physical Ångström coordinates
    spacing = box_size / (grid_size - 1)
    physical_coords = grid_coords.float() * spacing
    
    # Confidence values
    confidences = predicted_grid[is_peak]
    
    return physical_coords, confidences


# =====================================================================
# 5. EXECUTION & EVALUATION ON REAL UNSEEN PROTEIN
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # PDB Directory paths
    pdb_dir = "./pdb_data"
    ubiquitin_path = os.path.join(pdb_dir, "1ubq.cif") # Training Protein (Ubiquitin)
    crambin_path = os.path.join(pdb_dir, "1crn.cif")     # Evaluation Protein (Crambin)
    
    print("Loading real protein coordinate datasets...")
    train_coords = load_coords_biotite(ubiquitin_path)
    test_coords = load_coords_biotite(crambin_path)
    
    print(f"  Loaded Train: {os.path.basename(ubiquitin_path)} | CA Atoms: {len(train_coords)}")
    print(f"  Loaded Test:  {os.path.basename(crambin_path)} | CA Atoms: {len(test_coords)}")
    
    # Initialize the 3D CNN
    model = Tiny3DCNN()
    
    # Train model on Ubiquitin
    train_model(model, train_coords, num_epochs=120)
    
    # Evaluate model on the completely unseen Crambin structure!
    print("\nEvaluating on unseen Crambin structure...")
    model.eval()
    
    with torch.no_grad():
        # Rasterize Crambin (no noise during testing)
        test_in, centered_test_coords = rasterize_to_3d_grid(test_coords, sigma=2.0, noise_std=0.0)
        
        # Predict 3D density map: Shape [1, 1, 32, 32, 32]
        x_test = test_in.unsqueeze(0).unsqueeze(0)
        pred_map = model(x_test)
        
        # Resolve predicted discrete 3D coordinates
        pred_coords, confidences = find_peaks_3d(pred_map, threshold=0.35)
        
    print("\n" + "="*50)
    print(" 3D REAL PROTEIN EVALUATION RESULTS ")
    print("="*50)
    print(f"Ground Truth Crambin C-alpha coordinates (first 5 of {len(centered_test_coords)} atoms):")
    for i, coord in enumerate(centered_test_coords[:5]):
        print(f"  Atom {i+1}: ({coord[0].item():.2f}, {coord[1].item():.2f}, {coord[2].item():.2f})")
        
    print(f"\nModel Predicted Peak Coordinates (first 5 of {len(pred_coords)} resolved):")
    if len(pred_coords) == 0:
        print("  No peaks resolved!")
    for i, (coord, conf) in enumerate(zip(pred_coords[:5], confidences[:5])):
        print(f"  Peak {i+1}: ({coord[0].item():.2f}, {coord[1].item():.2f}, {coord[2].item():.2f}) | Conf: {conf.item():.4f}")
        
    # Calculate recovery rate within a standard physical distance matching radius (2.0 Å)
    matched = 0
    for gt in centered_test_coords:
        if len(pred_coords) > 0:
            distances = torch.norm(pred_coords.float() - gt.float(), dim=1)
            min_dist, _ = torch.min(distances, dim=0)
            if min_dist.item() <= 2.0:
                matched += 1
                
    accuracy = (matched / len(centered_test_coords)) * 100
    print("-"*50)
    print(f"Overall 3D Coordinate Recovery: {accuracy:.1f}% ({matched}/{len(centered_test_coords)} atoms resolved within 2.0 Å)")
    print("="*50)
