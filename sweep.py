import wandb, pprint

sweep_config = {
    "method": "grid",
    "program": "train_wandb.py",      # <-- IMPORTANT
    "metric": {"name": "traj/ODE/mean", "goal": "minimize"},
    "parameters": {
        "n_steps":    {"values": [5_000, 10_000]},
        "lr":         {"values": [5e-5, 1e-4, 3e-4]},
        "batch_size": {"values": [64]},
        "alpha":      {"values": [0.1, 0.2]},
        "dyn_alpha":  {"values": [0.001, 0.005, 0.01, 0.05, 0.1]},
        "dyn_hidden": {"values": [2, 4, 6, 8]},
        "reg":        {"values": [0.0, 1e-5, 1e-6]},
    },
}
pprint.pprint(sweep_config)
wandb.login()
sweep_id = wandb.sweep(sweep_config, project="structureflow-grid", entity="StructureFlow")
print("NEW SWEEP:", sweep_id)