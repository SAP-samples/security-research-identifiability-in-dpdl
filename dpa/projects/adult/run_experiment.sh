#!/bin/bash
runs=5
for ((i=1;i<=runs;i++)); do
    python ./run_experiment.py -it=1 -ep=30 -l2=3.0 -ls -dlt=0.001 -lr=0.005 -bnd -id=$i
done
sudo /sbin/shutdown -h now