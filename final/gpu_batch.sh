#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:20:00
#SBATCH --job-name=hpcfinal_gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu

cd "$HOME/hpc/final" || exit 1

GPU_EXEC=$HOME/hpc/final/mcgpu
HYBRID_EXEC=$HOME/hpc/final/mchybrid

ITERATIONS=(100000 1000000 10000000 100000000)

GPU_OUT=gpu.txt
HYBRID_OUT=hybrid.txt

> "$GPU_OUT"
> "$HYBRID_OUT"

for N in "${ITERATIONS[@]}"; do
    echo "=== N_ITER = $N 50% ===" | tee -a "$HYBRID_OUT"
    $HYBRID_EXEC $N 3 0.5 >> "$HYBRID_OUT"

    echo "=== N_ITER = $N 60% ===" | tee -a "$HYBRID_OUT"
    $HYBRID_EXEC $N 3 0.6 >> "$HYBRID_OUT"

    echo "=== N_ITER = $N 70% ===" | tee -a "$HYBRID_OUT"
    $HYBRID_EXEC $N 3 0.7 >> "$HYBRID_OUT"

    echo "=== N_ITER = $N 90% ===" | tee -a "$HYBRID_OUT"
    $HYBRID_EXEC $N 3 0.9 >> "$HYBRID_OUT"
done
