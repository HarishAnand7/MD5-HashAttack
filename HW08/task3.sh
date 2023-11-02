#!/usr/bin/env bash

#SBATCH --job-name=q3.sort
#SBATCH --partition=instruction
#SBATCH --time=0-00:00:10
#SBATCH --cpus-per-task=20
#SBATCH --error=error-%j.txt
#SBATCH --output=output-%j.txt
#SBATCH --mem=20G
cd $SLURM_SUBMIT_DIR


g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
for i in {1..20};
do
        ./task3 1000000 $i 32
done
