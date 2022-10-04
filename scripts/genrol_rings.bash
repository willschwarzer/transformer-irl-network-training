#!/bin/bash
# wandb offline
nice -n 15 python generate_rollouts_new.py --out rings_1000 --model-dir ring_agents --max-threads 16 --non-linear --num-examples 100 --num-models 1000 --env rings --verbose 1 --process-data --single-move
# wandb sync --sync-all