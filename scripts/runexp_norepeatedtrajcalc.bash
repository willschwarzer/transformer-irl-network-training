#!/bin/bash
# wandb offline
nice -n 15 python predict_transformer_nonlinear.py --num-examples 460000 --save-to more_layers.parameters --permute-types --wandb-project sirl --no-trajectory-sigmoid --no-space-invaders --num-trajectory-layers 1 --num-state-layers 1 --batch-size 128 --no-repeat-trajectory-calculations -v
# wandb sync --sync-all
