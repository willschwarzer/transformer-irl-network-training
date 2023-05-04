#!/bin/bash
# wandb offline
nice -n 15 python train_agents_will.py --env rings --num-timesteps 400000 --save-dir data/ring_agents_no_jax_mm --num-envs 100000 --alg PPO --max-threads 16 --no-single-move --no-verbose # --wandb-int
# wandb sync --sync-all