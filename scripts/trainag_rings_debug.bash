#!/bin/bash
# wandb offline
nice -n 15 python train_agents_will.py --env rings --num-timesteps 1000 --save-dir ring_agents_debug --num-envs 100 --alg PPO --max-threads 8 # --wandb-int
# wandb sync --sync-all