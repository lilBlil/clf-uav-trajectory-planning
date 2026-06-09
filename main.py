"""
main.py
=======
Entry point for the CLF-aided multi-UAV trajectory planning reproduction.

Usage:
    python main.py                 # full run with paper-style defaults
    python main.py --quick         # fast smoke test (small budget)
    python main.py --episodes 1200 # custom training budget
    python main.py --no-verbose    # silence per-update logging

Outputs (figures + CSV) are written to ./outputs/.
"""
from __future__ import annotations

import argparse

from src.config import Config
from experiments.run_experiment import run


def parse_args():
    p = argparse.ArgumentParser(description="CLF-aided multi-UAV MAPPO planner")
    p.add_argument("--quick", action="store_true",
                   help="fast smoke test (few episodes, coarse grid)")
    p.add_argument("--episodes", type=int, default=None,
                   help="override max training episodes Epi_max")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--no-verbose", action="store_true",
                   help="disable per-update logging")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.quick:
        cfg.voxel_size = 30.0
        cfg.epi_max = 80
        cfg.rollouts_per_update = 4
        cfg.max_steps = 40
    if args.episodes is not None:
        cfg.epi_max = args.episodes
    if args.seed is not None:
        cfg.seed = args.seed
    run(cfg=cfg, verbose=not args.no_verbose)


if __name__ == "__main__":
    main()
