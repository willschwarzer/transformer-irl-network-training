#!/bin/bash
# wandb offline
nice -n 15 python generate_rollouts_new.py --out rings_dummy --model-dir ring_agents_dummy --max-threads 2 --non-linear --num-examples 10 --num-models 4 --env rings --verbose 1
# wandb sync --sync-all
