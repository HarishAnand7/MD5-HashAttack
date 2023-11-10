#!/usr/bin/env zsh
#SBATCH --job-name=task3
#SBATCH --partition=instruction
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-00:00:10
#SBATCH --output=task3.out
#SBATCH --error=task3.err

cd $SLURM_SUBMIT_DIR

module load mpi/mpich/4.0.2

mpicxx task3.cpp -Wall -O3 -o task3

for n in {1..25};
do  
    srun -n 2 task3 $((2**n))
done

