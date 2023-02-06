#!/bin/sh

#SBATCH --job-name=gedholo
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=6:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=6144
#SBATCH --mail-type=END
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python main.py