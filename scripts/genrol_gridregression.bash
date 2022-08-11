#!/bin/bash
# wandb offline
nice -n 15 python generate_rollouts_new.py --out grid_regression_with_weights -ne 10000 -nm 10 -mt 10 -v 1 -cr
# wandb sync --sync-all
