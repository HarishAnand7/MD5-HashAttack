#!/usr/bin/env zsh
#SBATCH --job-name=task2
#SBATCH --partition=instruction
#SBATCH --time=0-00:00:10
#SBATCH --output=task2.out
#SBATCH --error=task2.err
#SBATCH --nodes=2 --cpus-per-task=20 --ntasks-per-node=1

cd $SLURM_SUBMIT_DIR

mpicxx task2.cpp reduce.cpp -Wall -O3 -o task2 -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec
n=10000000
for t in {1..20};
do
srun -n 2 --cpu-bind=none ./task2 $n $t
  echo
done
