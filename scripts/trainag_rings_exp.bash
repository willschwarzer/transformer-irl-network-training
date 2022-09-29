#!/bin/bash
# wandb offline
nice -n 15 python train_agents_will.py --env rings --num-timesteps 1000000 --save-dir ring_agents_multi_move --num-envs 10 --alg PPO --max-threads 10 --no-single-move --wandb-int
# wandb sync --sync-all