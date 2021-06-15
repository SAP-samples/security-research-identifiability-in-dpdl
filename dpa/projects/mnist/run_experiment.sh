#!/bin/bash
runs=250
for ((i=1;i<=runs;i++)); do
    python ./run_experiment.py -it=1 -ep=30 -l2=3.0 -ls -dlt=0.01 -lr=0.005 -bnd -mnst -id=$i
done
sudo /sbin/shutdown -h now