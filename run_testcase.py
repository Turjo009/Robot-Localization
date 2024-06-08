import numpy as np

def load_and_test_maps(file_path):
    """Loads maps from an NPZ file and performs basic tests.

    Args:
        file_path: Path to the .npz file containing the maps.
    """

    with np.load(file_path) as data:
        maps = [data[key] for key in data]  # Load all arrays into a list
    
    # Basic Tests
    num_maps = len(maps)
    print(f"Loaded {num_maps} maps.")
    
    for i, map in enumerate(maps):
        print(f"\nMap {i + 1}:")
        print(map)

        # Additional Tests (examples):
        # Check if all values are probabilities (between 0 and 1)
        if not np.all((map >= 0) & (map <= 1)):
            print("Warning: Map contains values outside the probability range [0, 1].")

        # Check if probabilities sum to 1 for each cell (optional)
        row_sums = map.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            print("Warning: Probabilities don't sum to 1 for some rows.")

        # ... other tests based on your specific requirements ...

if __name__ == "__main__":
    file_path = "op10.npz"  # Update if your file has a different name
    load_and_test_maps(file_path)
