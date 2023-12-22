#!/usr/bin/env bash

set -e

export DEBIAN_FRONTEND=noninteractive

echo -n "Please enter the path for the isaacgym directory (e.g. ~/isaacgym): "
read IsaacGymPath

cd $IsaacGymPath

git clone git@github.com:piratax007/aerial_gym_simulator.git

cp aerial_gym_simulator/docker/Dockerfile docker

bash docker/build.sh

cp aerial_gym_simulator/docker/run_isaac_plus_aerial.sh docker

cd ~

rm install_using_docker.sh

