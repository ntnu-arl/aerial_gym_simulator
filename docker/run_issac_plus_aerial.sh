#!/bin/bash
docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network=host --gpus=all --name=isaacgym_container achilleas2942/issac-aerial-gym:latest /bin/bash
