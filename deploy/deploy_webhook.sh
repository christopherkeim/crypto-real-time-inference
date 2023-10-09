#! /bin/bash

# Deploys the webhook binary if it is not currently up
if ( curl -X POST http://127.0.0.1:10000/infra/your-deploy-key-here > /dev/null )
then
  echo "Webhook server is running on http://127.0.0.1:10000/infra/ 🟢"
else
  echo "Deploying webhook server at http://127.0.0.1:10000/infra/ 🦫 🚀"
  deploy/webhook
fi
