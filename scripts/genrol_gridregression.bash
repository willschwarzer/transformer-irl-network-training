#!/bin/bash
# wandb offline
nice -n 15 python generate_rollouts_new.py --out grid_regression_with_weights -ne 100
# wandb sync --sync-all
