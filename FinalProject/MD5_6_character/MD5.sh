#!/usr/bin/env zsh
#SBATCH --job-name=6character
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:2
#SBATCH --error=errorq-%j.txt
#SBATCH --output=outputq-%j.txt


cd $SLURM_SUBMIT_DIR


nvcc MD5_GPU.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o MD5_GPU
./MD5_GPU

