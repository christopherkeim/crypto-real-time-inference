#! /bin/bash

# wandb API key
if [ -n "$WANDB_API_KEY" ]
then 
  echo "WANDB_API_KEY already set."
else
  echo "Setting WANDB_API_KEY environment variable"
  export WANDB_API_KEY="your-api-key"
fi 

# wandb entity name
if [ -n "$WANDB_ENTITY" ]
then 
  echo "WANDB_ENTITY already set."
else
  echo "Setting WANDB_ENTITY environment variable"
  export WANDB_ENTITY="your-entity-name"
fi 

# wandb project name
if [ -n "$WANDB_PROJECT" ]
then 
  echo "WANDB_PROJECT already set."
else
  echo "Setting WANDB_PROJECT environment variable"
  export WANDB_PROJECT="your-project-name"
fi 
