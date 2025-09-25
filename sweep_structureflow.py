import wandb
from src.models.components.sf2m_ngm1 import SF2MNGM
from src.datamodules.grn_datamodule import TrajectoryStructureDataModule

def main():
    with wandb.init(project="structureflow-grid"):
        wandb.define_metric("trainer/step")
        wandb.define_metric("*", step_metric="trainer/step")
        cfg = wandb.config

        dm = TrajectoryStructureDataModule()
        model = SF2MNGM(
            datamodule=dm,
            T=5, sigma=1.0, dt=0.2,
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

if __name__ == "__main__":
    main()