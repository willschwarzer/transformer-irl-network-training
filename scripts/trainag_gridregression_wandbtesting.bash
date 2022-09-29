#!/bin/bash
# wandb offline
nice -n 15 python train_agents_will.py --save-dir=none --max-threads=1 --num-timesteps=5000000 --wandb-int
# wandb sync --sync-all
