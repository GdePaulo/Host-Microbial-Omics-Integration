#!/bin/sh

#SBATCH --job-name=holo-ge
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=8:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=4500
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python main.py aak_ge