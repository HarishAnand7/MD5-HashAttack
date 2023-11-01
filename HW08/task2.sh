#!/usr/bin/env bash

#SBATCH --job-name=q2.convol
#SBATCH --partition=instruction
#SBATCH --time=0-00:00:10
#SBATCH --cpus-per-task=20
#SBATCH --error=error-%j.txt
#SBATCH --output=output-%j.txt
#SBATCH --mem=20G
cd $SLURM_SUBMIT_DIR


g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp
for i in {1..20};
do
    ./task2 1024 $i
done
