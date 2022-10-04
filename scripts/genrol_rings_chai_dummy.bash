#!/bin/bash
# wandb offline
nice -n 15 python generate_rollouts_new.py --out rings_chai_dummy --model-dir ring_agents --max-threads 16 --non-linear --num-examples 100 --num-models 10 --env rings --verbose 1 --process-data --single-move --chai-rollouts
# wandb sync --sync-all