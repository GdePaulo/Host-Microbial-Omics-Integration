#!/bin/sh

dos2unix sub_job_ge.sbatch
sbatch sub_job_ge.sbatch
dos2unix sub_job_gen.sbatch
sbatch sub_job_gen.sbatch