#! /bin/bash

# Replace "your-entity-name" on LINE 16 with the name of your public W&B model registry entity name
#  
# This script sets up the WANDB_ENTITY environment variable necessary to 
# download models from a public Weights & Biases model registry. 
# 
# For use, this script should be sourced into your current shell instance with `source deploy/environment.sh`
# or `./deploy/environment.sh` - the environment variable will be dropped after the shell session terminates.


# wandb entity name
if [ -n "$WANDB_ENTITY" ]
then 
  echo "WANDB_ENTITY already set."
else
  echo "Setting WANDB_ENTITY environment variable 🔧"
  # Replace with your entity name
  export WANDB_ENTITY="your-entity-name"
fi
