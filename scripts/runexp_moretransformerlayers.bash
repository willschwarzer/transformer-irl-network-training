#!/bin/bash
# wandb offline
nice -n 15 python predict_transformer_nonlinear.py --num-examples 460000 --save-to more_layers.parameters --permute-types --wandb-project sirl --no-trajectory-sigmoid --no-space-invaders --num-transformer-layers 5 --batch-size 2 -v
# wandb sync --sync-all
