#!/bin/sh

#SBATCH --job-name=gedholo
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:10:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=6144
#SBATCH --mail-type=END
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun -n1 --mem-per-cpu 1500MB --exclusive python main.py tcma_gen &
srun -n1 --mem-per-cpu 1500MB --exclusive python main.py aak_ge &
srun -n1 --mem-per-cpu 1500MB --exclusive python main.py tcma_gen_aak_ge &
srun -n1 --mem-per-cpu 1500MB --exclusive python main.py tcma_gen_aak_ge_ae &
wait