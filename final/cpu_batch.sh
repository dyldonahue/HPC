#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH --job-name=hpcfinal_cpu
#SBATCH --partition=courses
#SBATCH --nodelist=c0745

cd "$HOME/hpc/final" || exit 1

# Paths to executables
BASE_EXEC=$HOME/hpc/final/mcbase
OPT_CPU_EXEC=$HOME/hpc/final/mccpu

# N_ITER values to test
ITERATIONS=(100000 1000000 10000000 100000000)

# Output files
BASE_OUT=base.txt
CPU_OPT_OUT=cpu.txt

# Clear existing output files
> "$BASE_OUT"
> "$CPU_OPT_OUT"

for N in "${ITERATIONS[@]}"; do
    echo "=== N_ITER = $N ===" | tee -a "$BASE_OUT"
    $BASE_EXEC $N 3 >> "$BASE_OUT"

    echo "=== N_ITER = $N ===" | tee -a "$CPU_OPT_OUT"
    $OPT_CPU_EXEC $N 3 >> "$CPU_OPT_OUT"
done
