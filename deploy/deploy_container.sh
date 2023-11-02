#! /bin/bash

# Spin up the Prediction server if the container is not running; OR if the image on disk 
# isn't up to date, pull the new image, tear down the running container, and spin up the
# new container
sudo docker compose -f deploy/prediction-docker-compose.yml up --detach --pull predict