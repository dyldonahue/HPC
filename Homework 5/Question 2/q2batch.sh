#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=hw5q2
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu

VALUES=(16 32 64)
OUTPUT_FILE="hw5q2.out"

# Clear the output file before writing to it
> $OUTPUT_FILE

for N in "${VALUES[@]}"; do
    for MODE in 0 1; do
        echo "Running MODE=$MODE, N=$N" >> $OUTPUT_FILE
        ./hpc/hw5/q1b $MODE $N >> $OUTPUT_FILE 2>&1
        echo -e "\n------------------------\n" >> $OUTPUT_FILE
    done
done
