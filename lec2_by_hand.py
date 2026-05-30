import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import biotite.structure.io.pdbx as pdbx

EPOCHS=500
LR=0.002
GRID_SIZE=32
BOX_SIZE=32.0
THRESHOLD=0.35
PDB_DIR = "./pdb_data"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using {DEVICE} as device")

torch.manual_seed(42)

def load_coords(path):
    cif_file=pdbx.CIFFile.read(path)
    atoms: Any = pdbx.get_structure(cif_file, model=1)
    valid_atoms = atoms[(atoms.res_name != "HOH") & (atoms.atom_name == "CA")]
    return torch.tensor(valid_atoms.coord, dtype=torch.float32)

def kde(coords, sigma, noise=0.0):
    device = coords.device
    # Center coords
    center = coords.mean(dim=0)
    centered_coords = coords - center + (BOX_SIZE/2.0)

    # Set up grid
    ticks = torch.linspace(0.0, BOX_SIZE, GRID_SIZE, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(ticks,ticks,ticks, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1,3) # [32768, 3]

    # Compute total distance between all grid voxels and atoms
    g2 = torch.sum(grid ** 2, dim=-1, keepdim=True) # [32768, 1]
    c2 = torch.sum(centered_coords ** 2, dim=-1, keepdim=True).t() # [1, N]
    total_dist = torch.clamp(g2 + c2 - 2.0 * torch.matmul(grid, centered_coords.t()),min=0.0)

    # Generate Gaussian density grid, apply noise
    density_grid = torch.clamp(torch.sum(torch.exp(-total_dist/ (2 * (sigma ** 2))), dim=-1), max=1)
    density_grid += torch.clamp(torch.rand_like(density_grid) * noise, min=0.0)
    density_grid = density_grid.view(GRID_SIZE, GRID_SIZE, GRID_SIZE)

    return density_grid, centered_coords

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return self.net(x)

def mse_loss(pred, target):
    return F.mse_loss(pred, target)

def train_model(model, train_coords):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range (1, EPOCHS+1):
        clean_target, _ = kde(train_coords, sigma=1.0)
        noisy_in, _ = kde(train_coords, sigma=2.0, noise=0.04)

        x = noisy_in.unsqueeze(0).unsqueeze(0)
        y = clean_target.unsqueeze(0).unsqueeze(0)

        pred = model(x)
        loss = mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.3f}")

def find_peaks(predicted_grid):
    max_pooled = F.max_pool3d(predicted_grid, kernel_size=3, stride=1, padding=1)
    peak_indicies = torch.nonzero((predicted_grid == max_pooled) & (predicted_grid > THRESHOLD))
    grid_coords = peak_indicies[:, 2:]
    spacing = BOX_SIZE / (GRID_SIZE - 1)
    return grid_coords.float() * spacing

if __name__ == "__main__":
    train_coords = load_coords(os.path.join(PDB_DIR, "1ubq.cif")).to(DEVICE)
    test_coords = load_coords(os.path.join(PDB_DIR, "1crn.cif")).to(DEVICE)

    model = CNN().to(DEVICE)
    train_model(model, train_coords)

    model.eval()
    with torch.no_grad():
        test_in, centered_test_coords = kde(test_coords, sigma=2.0)
        x_test = test_in.unsqueeze(0).unsqueeze(0)
        pred_map = model(x_test)
        pred_coords = find_peaks(pred_map)

    matched = 0
    for gt in centered_test_coords:
        if len(pred_coords) > 0:
            distances = torch.norm(pred_coords.float() - gt.float(), dim=1)
            if torch.min(distances).item() <= 2.0:
                matched += 1

    accuracy = (matched / len(centered_test_coords)) * 100
    print(f"Overall 3D Coordinate Recovery: {accuracy:.1f}%")
