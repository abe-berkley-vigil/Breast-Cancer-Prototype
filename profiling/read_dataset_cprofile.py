from pathlib import Path
import pstats

# Define paths
base_dir = Path(__file__).resolve().parents[1]
profile_path = base_dir / "reports" / "profiling" / "dataset.prof"
output_path = base_dir / "reports" / "profiling" / "dataset_profile.txt"

# Generate and write stats to a text file
with open(output_path, "w") as f:
    stats = pstats.Stats(str(profile_path), stream=f)
    stats.strip_dirs().sort_stats("cumtime").print_stats()

print(f"Saved output to {output_path}")