#!/usr/bin/env zsh
#SBATCH --job-name=task2
#SBATCH --partition=instruction
#SBATCH --time=0-00:00:10
#SBATCH --output=task2.out
#SBATCH --error=task2.err
#SBATCH --cpus-per-task=10

cd $SLURM_SUBMIT_DIR

g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec

for t in {1..10};
do
./task2 1000000 $t
done
