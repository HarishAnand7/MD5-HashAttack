#!/usr/bin/env zsh
#SBATCH --job-name=FirstSlurm
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:00:10
#SBATCH --output=FirstSlum-%j.out
#SBATCH --error=FirstSlurm-%j.err


##cd $SLURM_SUBMIT_DIR

echo "Hostname: "
hostname
~
