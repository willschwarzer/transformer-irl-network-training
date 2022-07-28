#!/bin/bash
# wandb offline
nice -n 15 python predict_transformer_nonlinear.py --num-examples 1010000 --save-to more_layers.parameters --wandb-project sirl --no-trajectory-sigmoid --no-space-invaders --trajectory-hidden-size 1 --num-trajectory-layers 1 --state-hidden-size 2048 --num-state-layers 2 --batch-size 256 --no-repeat-trajectory-calculations --ground-truth-weights -v --use-shuffled
# wandb sync --sync-all
