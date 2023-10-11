#! /bin/bash

# Modify DOCKER_HUB_DEPLOY_KEY value in `environment.sh` (line 19) before running this script.
#
# Deploys the crypto-real-time-inference application onto a Ubuntu or Debian-based
# server.
# 
# Note that each script called is idempotent.

# Setup the deployment environment
bash deploy/setup_deploy_deb.sh

# Set environment variables for deploy
source ./deploy/environment.sh

# Spin up the Prediction server
bash deploy/deploy_container.sh

# Spin up the Webhook server
bash deploy/deploy_webhook.sh
