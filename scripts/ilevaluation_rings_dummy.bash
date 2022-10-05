#!/bin/bash
# wandb offline
nice -n 15 python IL_evaluation.py --data rings_chai_dummy --weights rings_chai_dummy_weights.npy --model ground_truth_phi_test.parameters --num-rollouts-per-agent 100 --single-move --agent-directory ring_agents --rl-its 1 --il-its 16384
# wandb sync --sync-all