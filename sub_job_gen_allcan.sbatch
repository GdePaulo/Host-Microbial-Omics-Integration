#!/bin/sh

#SBATCH --job-name=holo-gen
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=14:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=3500
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python main.py tcma_gen