import pandas as pd
from pathlib import Path

# change this to the directory that holds all dataset folders
root = Path("results2")

all_metrics = pd.concat(
    [pd.read_csv(p) for p in root.rglob("metrics_summary.csv")],
    ignore_index=True,
)

all_metrics.to_csv("metrics_summary_all_datasets_1.csv", index=False)
print(all_metrics)