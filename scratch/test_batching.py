import torch
import sys
import os

# Include current directory in path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from lec2 import UNet3D, BatchedPeakFinder3D

def test_reusable_pipeline():
    print("="*60)
    print(" TESTING REUSABLE BATCHED PIPELINE ")
    print("="*60)
    
    # Configure shapes
    B = 3          # Batch size of 3
    C = 1          # Single channel (density maps)
    X = Y = Z = 20 # 20x20x20 spatial grid
    
    print(f"Creating mock density grids of shape: [{B}, {C}, {X}, {Y}, {Z}]")
    mock_input = torch.full((B, C, X, Y, Z), -5.0)
    
    # 1. Test UNet3D shape consistency
    print("\n[1] Testing UNet3D shape propagation...")
    unet = UNet3D(in_channels=1, out_channels=1, init_features=8)
    unet.eval()
    
    with torch.no_grad():
        output = unet(mock_input)
        
    print(f"    Input shape:  {mock_input.shape}")
    print(f"    Output shape: {output.shape}")
    assert output.shape == mock_input.shape, "Error: UNet3D output shape mismatch!"
    print("    -> UNet3D shape assertion passed!")

    # 2. Test BatchedPeakFinder3D extraction and shape correctness
    print("\n[2] Testing BatchedPeakFinder3D coordinate extraction...")
    max_peaks = 12
    peak_finder = BatchedPeakFinder3D(threshold=0.1, kernel_size=3, max_peaks=max_peaks, box_size=10.0)
    
    # Setup some deterministic high peaks in mock_input
    # Batch 0: 2 high peaks
    mock_input[0, 0, 5, 5, 5] = 2.5
    mock_input[0, 0, 10, 10, 10] = 3.0
    
    # Batch 1: 1 high peak
    mock_input[1, 0, 15, 15, 15] = 4.0
    
    # Batch 2: 0 peaks (all negative/below threshold)
    mock_input[2] = -5.0
    
    with torch.no_grad():
        coords, values, mask = peak_finder(mock_input)
        
    print(f"    Coordinates shape: {coords.shape}  | Expected: [{B}, {max_peaks}, 3]")
    print(f"    Values shape:      {values.shape}  | Expected: [{B}, {max_peaks}]")
    print(f"    Mask shape:        {mask.shape}  | Expected: [{B}, {max_peaks}]")
    
    assert coords.shape == (B, max_peaks, 3), "Error: Coordinates shape mismatch!"
    assert values.shape == (B, max_peaks), "Error: Values shape mismatch!"
    assert mask.shape == (B, max_peaks), "Error: Mask shape mismatch!"
    print("    -> Peak finder shape assertions passed!")
    
    # Check specific batch outputs
    print("\n[3] Verifying extracted peak statistics:")
    for b in range(B):
        num_valid = mask[b].sum().item()
        print(f"    Batch {b} -> Found {num_valid} valid peaks (Max {max_peaks})")
        if num_valid > 0:
            print(f"      Top coordinate: {coords[b, 0].tolist()}")
            print(f"      Top confidence: {values[b, 0].item():.4f}")
            
    # Check Batch 0 values
    assert mask[0, 0].item() is True, "Batch 0 peak 1 should be valid"
    assert mask[0, 1].item() is True, "Batch 0 peak 2 should be valid"
    assert mask[0, 2].item() is False, "Batch 0 peak 3 should be padded"
    
    # Check Batch 2 values (no peaks above threshold)
    assert mask[2].sum().item() == 0, "Batch 2 should have 0 valid peaks"
    print("\n[4] All unit tests completed successfully!")
    print("="*60)

if __name__ == "__main__":
    test_reusable_pipeline()
