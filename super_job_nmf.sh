#!/bin/sh

dos2unix sub_job_ge_nmf.sbatch
sbatch sub_job_ge_nmf.sbatch
dos2unix sub_job_gen_nmf.sbatch
sbatch sub_job_gen_nmf.sbatch 
dos2unix sub_job_genge_nmf.sbatch
sbatch sub_job_genge_nmf.sbatch 