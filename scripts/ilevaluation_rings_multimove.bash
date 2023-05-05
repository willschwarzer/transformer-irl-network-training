#!/bin/bash
# wandb offline
nice -n 15 python IL_evaluation.py --data rings_chai_multimove100 --weights rings_chai_multimove100_weights.npy --model ground_truth_phi_test.parameters --num-rollouts-per-agent 10000 --no-single-move --agent-directory ring_agents_multi_move --rl-its 400000 --il-its 1000000 --max-threads 10
# wandb sync --sync-all