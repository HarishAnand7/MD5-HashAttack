#!/usr/bin/env zsh
#SBATCH --job-name=q1.thrust
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:00:10
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --error=error_thrust-%j.txt
#SBATCH --output=output_thrust-%j.txt
#SBATCH --mem=20G
cd $SLURM_SUBMIT_DIR

nvcc task1_thrust.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1_thrust

for i in {10..20};
do
   ./task1_thrust $((2**i))
done
