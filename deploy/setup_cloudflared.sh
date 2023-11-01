#!/bin/bash

#-----------------------------------------------------------------------------------------------------------#
# This script sets up a deployment environment on an Ubuntu 20.04/22.04
#
# Documentation:
# https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-local-tunnel/
#
#-----------------------------------------------------------------------------------------------------------#

if ( which cloudflared > /dev/null )
then
  echo "cloudflared installed 🟢"
else
# Add Cloudflare’s package signing key
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null

# Add Cloudflare’s apt repo to your apt repositories
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/cloudflared.list

# Update repositories and install cloudflared
sudo apt-get update && sudo apt-get install cloudflared
fi