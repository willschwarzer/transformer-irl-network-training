#!/bin/bash
# wandb offline
nice -n 15 python IL_evaluation.py --data rings_chai_multimove100 --weights rings_chai_multimove100_weights.npy --model nshot_2.parameters --num-rollouts-per-agent 10000 --no-single-move --agent-directory ring_agents --rl-its 400 --adv-its 20000 --bc-epochs 1 --num-eval-episodes 1 --percentile-evaluation --agent-save-dir imitation_agents_dummy --num-trials=2
# wandb sync --sync-all