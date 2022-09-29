#!/bin/bash
# wandb offline
nice -n 15 python train_agents_will.py --env rings --num-timesteps 10000000 --save-dir ring_agents --num-envs 20 --alg PPO --wandb-int
# wandb sync --sync-all