#!/usr/bin/env python

from setuptools import find_packages, setup

install_requires = [
    "torch>=1.11.0",
    "matplotlib",
    "numpy",  # Due to pandas incompatibility
    "scipy",
    "scikit-learn",
    "torchdyn>=1.0.6",
    "pot",
    "torchdiffeq",
    "absl-py",
    "pandas>=2.2.2",
]

setup(
    name="src",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/user/project",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
