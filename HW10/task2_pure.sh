#!/usr/bin/env zsh
#SBATCH --job-name=task2_pure
#SBATCH --partition=instruction
#SBATCH --time=0-00:00:10
#SBATCH --output=task2_pure.out
#SBATCH --error=task2_pure.err
#SBATCH --nodes=2 --cpus-per-task=20 --ntasks-per-node=1

cd $SLURM_SUBMIT_DIR

g++ task2_pure_omp.cpp reduce.cpp -Wall -O3 -o task2_pure_omp -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec
n=20000000
for t in {1..20};
do
 ./task2_pure_omp $n $t
  echo
done
