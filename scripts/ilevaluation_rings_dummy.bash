#!/bin/bash
# wandb offline
nice -n 15 python IL_evaluation.py --data 
# wandb sync --sync-all