#!/bin/bash
rm -rf last_checkpoint
cp -r saved/latest last_checkpoint
echo "############### Last checkpoint back-up to last_checkpoint/. ##############"
# rm -rf saved/latest
sbatch singlenode.sh
