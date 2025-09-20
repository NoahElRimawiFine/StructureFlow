import os, wandb
from sweep_structureflow import train 

SWEEP = "StructureFlow/structureflow-grid/lt6thzsj"

if __name__ == "__main__":
    wandb.agent(SWEEP, function=train)