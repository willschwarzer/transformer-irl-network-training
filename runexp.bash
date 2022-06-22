#!/bin/bash
# wandb offline
nice -n 15 python predict_transformer_nonlinear.py --num-examples 460000 --save-to proper_split_mlp.parameters --mlp -v
# wandb sync --sync-all
