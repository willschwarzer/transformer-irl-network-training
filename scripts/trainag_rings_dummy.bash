#!/bin/bash
# wandb offline
nice -n 15 python train_agents_will.py --env rings --num-timesteps 1000 --save-dir ring_agents_dummy --num-envs 20 --alg PPO --wandb-int --max-threads 2
# wandb sync --sync-all