#!/usr/bin/env zsh
#SBATCH --job-name=task1
#SBATCH --partition=instruction
#SBATCH --time=0-00:00:10
#SBATCH --output=task1.out
#SBATCH --error=task1.err
#SBATCH -N 1 -c 1

cd $SLURM_SUBMIT_DIR


g++ task1.cpp optimize.cpp -Wall -O3 -std=c++17 -o task1 -fno-tree-vectorize
./task1 1000000

g++ task1.cpp optimize.cpp -Wall -O3 -std=c++17 -o task1 -march=native -fopt-info-vec -ffast-math
./task1 1000000
