#!/usr/bin/env zsh
#SBATCH --job-name=task1
#SBATCH --partition=instruction
#SBATCH --time=0-00:00:10
#SBATCH --output=task1.out
#SBATCH --error=task1.err
#SBATCH --cpus-per-task=10

cd $SLURM_SUBMIT_DIR


g++ task1.cpp cluster.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

for t in {1..10};
do
./task1 5040000 $t
done
