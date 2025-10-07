import itertools, subprocess, sys

backbones = ["dyn-BF", "dyn-TF", "dyn-SW", "dyn-CY", "dyn-LL"]
subsets = ["all"]

base_cmd = [
    sys.executable,
    "tigon_baseline.py",
    "--niters",
    "100",
    "--timepoints",
    "0,0.25,0.5,0.75,1.0",
    "--hidden-dim",
    "16",
]

for bb, sub in itertools.product(backbones, subsets):
    print(f"\n=== RUN {bb}  subset={sub} ===")
    cmd = base_cmd + ["--backbone", bb, "--subset", sub]
    print("running", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {cmd} → exit {e.returncode}")
        print(e.stderr)
        continue
