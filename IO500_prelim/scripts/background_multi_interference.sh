#/usr/bin/env bash

nohup ./run_multi_interference_levels.sh > multi_level_run.txt 2>&1 &
echo $! > multi_level_run_pid.txt