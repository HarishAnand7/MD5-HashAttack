#!/usr/bin/env zsh
#SBATCH --job-name=task1
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:00:10
#SBATCH --output=task1.out
#SBATCH --error=task1.err

cd $SLURM_SUBMIT_DIR

g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

for n in {1..20};
do
    ./task1 1024 $n
done
