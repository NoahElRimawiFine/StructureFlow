import pprint
import wandb
from src.models.components.sf2m_ngm1 import SF2MNGM
from src.datamodules.grn_datamodule import TrajectoryStructureDataModule

sweep_config = {
    "method": "grid",
    "metric": {"name": "traj/ODE/mean", "goal": "minimize"},
    "parameters": {
        "n_steps":      {"values": [5_000, 10_000]},
        "lr":           {"values": [5e-5, 1e-4, 3e-4]},
        "batch_size":   {"values": [64]},
        "alpha":        {"values": [0.1, 0.2]},
        "dyn_alpha":    {"values": [0.001, 0.005, 0.01, 0.05, 0.1]},
        "dyn_hidden":   {"values": [2, 4, 6, 8]},
        "reg":          {"values": [0.0, 1e-5, 1e-6]},
    },
}

pprint.pprint(sweep_config)
wandb.login()
sweep_id = wandb.sweep(sweep_config, project="structureflow-grid")  

def train():
    with wandb.init(project="structureflow-grid") as run:
        # Make all metrics use "trainer/step" as their x-axis
        wandb.define_metric("trainer/step")
        wandb.define_metric("*", step_metric="trainer/step")

        cfg = wandb.config

        datamodule = TrajectoryStructureDataModule()
        model = SF2MNGM(
            datamodule=datamodule,
            T=5,
            sigma=1.0,
            dt=0.2,
            batch_size=cfg.batch_size,
            alpha=cfg.alpha,
            reg=getattr(cfg, "reg", 0.0),
            dyn_hidden=cfg.dyn_hidden,
            dyn_alpha=cfg.dyn_alpha,
            correction_reg_strength=1e-3,
            n_steps=cfg.n_steps,
            lr=cfg.lr,
            device=None,
            GL_reg=0.04,
            knockout_hidden=100,
            score_hidden=[100, 100],
            correction_hidden=[64, 64],
        )

        model.train_model(skip_time=None)
