#!/bin/bash
# wandb offline
nice -n 15 python train_agents_will.py --env rings --num-timesteps 400000 --save-dir ring_agents --num-envs 10000 --alg PPO --max-threads 50 # --wandb-int
# wandb sync --sync-all