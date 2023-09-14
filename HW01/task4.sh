#!/usr/bin/env zsh
#SBATCH --job-name=FirstSlurm
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:00:10
#SBATCH --output=FirstSlurm-%j.out
#SBATCH --output=FirstSlurm-%j.err


##cd $SLURM_SUBMIT_DIR

hostname
