#!/bin/sh


dos2unix sub_job_ge.sbatch
sbatch sub_job_ge.sbatch
dos2unix sub_job_gen.sbatch
sbatch sub_job_gen.sbatch
dos2unix sub_job_genge.sbatch
sbatch sub_job_genge.sbatch
dos2unix sub_job_genge_ae.sbatch
sbatch sub_job_genge_ae.sbatch 
dos2unix sub_job_genge_nmf.sbatch
sbatch sub_job_genge_nmf.sbatch 