#!/bin/bash
# wandb offline
nice -n 15 python generate_rollouts_new.py --out rings_multimove --model-dir ring_agents_multi_move --max-threads 16 --non-linear --num-examples 100 --num-models 1000 --env rings --no-single-move --verbose 1 --process-data
# wandb sync --sync-all