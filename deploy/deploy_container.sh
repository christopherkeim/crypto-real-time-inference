#! /bin/bash

# Spin up the Prediction server
docker compose -f deploy/prediction-docker-compose.yml up --detach --pull predict