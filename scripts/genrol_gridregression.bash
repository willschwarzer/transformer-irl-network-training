#!/bin/bash
# wandb offline
nice -n 15 python generate_rollouts_new.py --out grid_regression_with_weights -ne 100 -nm 1000 -mt 10 -v 1
# wandb sync --sync-all
