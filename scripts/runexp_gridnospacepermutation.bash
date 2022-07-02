#!/bin/bash
# wandb offline
nice -n 15 python predict_transformer_nonlinear.py --num-examples 460000 --save-to no_empty_permute.parameters --permute-types --wandb-project sirl --no-trajectory-sigmoid --no-space-invaders -v
# wandb sync --sync-all
