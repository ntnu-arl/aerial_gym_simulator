#!/usr/bin/env bash
export AERIAL_GYM_DIR=$HOME/workspaces/aerial_gym_ws/aerial_gym_simulator/aerial_gym
python3 dce_nn_navigation.py --train_dir=$AERIAL_GYM_DIR/examples/dce_rl_navigation/selected_network --experiment=selected_network --env=test --obs_key="observations" --load_checkpoint_kind=best