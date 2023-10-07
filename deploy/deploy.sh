#! /bin/bash

# Modify THE WANDB_ENTITY value in `environment.sh` (line 16) before running this script.
#
# Deploys the crypto-real-time-inference application onto a Ubuntu or Debian-based
# server.
# 
# Note that each script called is idempotent.
# 
# Args (positional)
#   - 1) MODELS: selection value for "all", "nn", or "ml"


# Validate passed in MODELS value
if [ -n "$1" ]
then
  echo "Using $1 as MODELS selection 🟢"
  MODELS=$1
else
echo "Using default 'all' as MODELS selection"
  MODELS="all"
fi

# Setup the deployment environment
bash deploy/setup_deploy_deb.sh

# Set environment variables for model downloads
source ./deploy/environment.sh

# Download models on disk
poetry run python deploy/download_models_from_wandb.py -s $MODELS

# Spin up the prediction server
poetry run python src/server.py
