#!/usr/bin/env bash

set -e

export DEBIAN_FRONTEND=noninteractive

echo -n "Please enter the path for the isaacgym directory (e.g. ~/isaacgym): "
read IsaacGymPath

cd $IsaacGymPath

git clone git@github.com:ntnu-arl/aerial_gym_simulator.git

cp aerial_gym_simulator/docker/Dockerfile ../docker

cp aerial_gym_simulator/docker/run_isaac_plus_aerial.sh
