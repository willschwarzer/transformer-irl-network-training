#!/bin/bash
# wandb offline
nice -n 15 python generate_rollouts_new.py --out rings_chai_multimove100 --model-dir ring_agents_multi_move --max-threads 16 --non-linear --num-examples 10000 --num-models 100 --env rings --verbose 1 --process-data --no-single-move --chai-rollouts
# wandb sync --sync-all