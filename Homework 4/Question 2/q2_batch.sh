#!/bin/bash
#SBATCH --verbose
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=hwq2dylandonahue
#SBATCH --mem=100G
#SBATCH --partition=courses
$SRUN mpirun -mca btl_bas:wq:quUie_warn_component_unused 0 ~/hpc/hw4/q2/q2a
