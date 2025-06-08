import torch
from torch.profiler import profile, ProfilerActivity
from pathlib import Path
from breast_cancer_classification.modeling.train import main as train


# Define paths
base_dir = Path(__file__).resolve().parents[1]
output_path = base_dir / "reports" / "profiling" / "pytorch_train_trace.json"

# Profile the training process
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with torch.profiler.record_function("train_model_full"):
        train()
    prof.step()

# Display the profiling results
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Export the profiling results to a JSON file
prof.export_chrome_trace(str(output_path))