import json
import wandb
import sys

run_id = "9qnh0b9k"  # The 125m calibrated run we found earlier
# Actually we can just read the offline wandb logs to check W_rms_min or W_frob for the layers at step 0
