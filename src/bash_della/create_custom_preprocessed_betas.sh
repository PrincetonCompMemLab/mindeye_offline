#!/usr/bin/env bash
#SBATCH -t 200
#SBATCH --mail-user=rk1593@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name get_cluster_exemplars_exaggerate0
#SBATCH -c 2
#SBATCH -N 1


module load anaconda3/2021.5
conda activate rt_mindEye2
module load fsl/6.0.6.2 
python create_custom_preprocessed_betas.py