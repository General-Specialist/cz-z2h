import torch
import gemmi

def cif_to_density(filepath: str, element_filter: str | None = None) -> torch.Tensor:
    # Parse mmCIF into structure object
    structure: gemmi.Structure = gemmi.read_structure(filepath)

    # Flatten the hierarchical structure (Model -> Chain -> Residue -> Atom) to extract all atoms
    atoms: list[gemmi.Atom] = [atom for model in structure for chain in model for residue in chain for atom in residue]

    # Extract coords and element names into PyTorch tensor
    coords: list[list[float]] = [[a.pos.x, a.pos.y, a.pos.z] for a in atoms]
    coordinates: torch.Tensor = torch.tensor(coords, dtype=torch.float32)

    # We compute the center of mass using all atoms to keep physical grids consistent
    # when filtering by specific elements.
    center_of_mass: torch.Tensor = coordinates.mean(dim=0)

    if element_filter is not None:
        filtered_atoms: list[gemmi.Atom] = [a for a in atoms if a.element.name.strip().upper() == element_filter.strip().upper()]
        if not filtered_atoms:
            return torch.zeros((20, 20, 20))
        coords = [[a.pos.x, a.pos.y, a.pos.z] for a in filtered_atoms]
        coordinates = torch.tensor(coords, dtype=torch.float32)

    voxel_size: float = 0.5
    box_size: float = 10.0
    sigma: float = 0.8

    # Center molecule
    coords_centered: torch.Tensor = coordinates - center_of_mass + (box_size / 2.0)

    # Define the 3D spatial grid ticks
    grid_size: int = int(box_size / voxel_size)
    ticks: torch.Tensor = torch.linspace(0, box_size, grid_size)


    grid_x: torch.Tensor
    grid_y: torch.Tensor
    grid_z: torch.Tensor
    grid_x, grid_y, grid_z = torch.meshgrid(ticks, ticks, ticks, indexing='ij')
    grid: torch.Tensor = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    grid_expanded: torch.Tensor = grid.unsqueeze(-2)
    coords_expanded: torch.Tensor = coords_centered[None, None, None, :, :]
    sq_distances: torch.Tensor = torch.sum((grid_expanded - coords_expanded) ** 2, dim=-1)
    atom_densities: torch.Tensor = torch.exp(-sq_distances / (2 * (sigma ** 2)))
    density_map: torch.Tensor = atom_densities.sum(dim=-1)
    return density_map

if __name__ == "__main__":
    print(cif_to_density("data.cif"))
