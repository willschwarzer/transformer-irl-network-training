#!/bin/bash
# wandb offline
nice -n 15 python IL_evaluation.py --data rings_chai_multimove100 --weights rings_chai_multimove100_weights.npy --model nshot_2.parameters --num-rollouts-per-agent 100 --no-single-move --agent-directory ring_agents --rl-its 400000 --adv-its 400000 --bc-epochs 100 --num-eval-episodes 100 --percentile-evaluation --agent-save-dir imitation_agents_fewer_dems --methods sirl
# wandb sync --sync-all