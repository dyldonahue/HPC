#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=hw5q1
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu

VALUES=(4096 32768 262144 2097152 8388608)

# Loop over arguments
for N in "${VALUES[@]}"; do
    OUTPUT_FILE="hw5q1_${N}.out"  
    ./hpc/hw5/q1b $N > $OUTPUT_FILE 2>&1
done



