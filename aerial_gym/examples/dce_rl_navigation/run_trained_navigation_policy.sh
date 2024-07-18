#!/usr/bin/env bash

python3 dce_nn_navigation.py --train_dir=$(pwd)/selected_network --experiment=selected_network --env=test --obs_key="observations" --load_checkpoint_kind=best
