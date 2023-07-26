#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath $0)")

PYTHON=~/anaconda3/envs/rl/bin/python

$PYTHON $SCRIPT_DIR/server/arctic_server.py