#!/bin/sh

#SBATCH --job-name=holo-ge
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=2000
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python main.py aak_ge