#!/bin/sh

#SBATCH --job-name=holo-genge-nmf
#SBATCH --partition=general 
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2000
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python main.py tcma_gen_aak_ge_nmf