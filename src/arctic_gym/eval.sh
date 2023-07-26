#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath $0)")
PROJECT_DIR=$(dirname $(dirname "$(realpath $SCRIPT_DIR)"))

export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.1
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

PYTHON=~/anaconda3/envs/rl/bin/python

$PYTHON $SCRIPT_DIR/arctic_env/env_debugger.py
##$PYTHON $SCRIPT_DIR/eval/n_routes.py