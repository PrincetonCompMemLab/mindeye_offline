#!/bin/bash
#SBATCH --job-name=ses-04
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1              
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1       # Set to equal gres=gpu:#!
#SBATCH --cpus-per-task=40      # 40 / 80 / 176 distributed across node
#SBATCH --time=00:45:00         # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err     # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --no-requeue
#SBATCH --array=0               # 0-9
#SBATCH --mail-type=END
#SBATCH --mail-user=rsiyer@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is ${SLURM_ARRAY_JOB_ID}"
echo "My SLURM_ARRAY_TASK_ID is ${SLURM_ARRAY_TASK_ID}"
echo "Executing on the machine: $(hostname)"

module load anaconda3/2023.3
conda activate rt_mindEye2

# verify these variables before submitting
# ---
sub=sub-001
session=ses-04
split=MST  # MST train/test split, alternative would be train on non-repeats and test on images that repeat (split=orig) 
model_name=${sub}_${session}_bs24_MST_rishab_${split}split
# --- 

jupyter nbconvert recon_inference-multisession-simple.ipynb --to python && \
python recon_inference-multisession-simple.py --model_name=${model_name} --subj=1 --no-blurry_recon --use_prior --hidden_dim=1024 --n_blocks=4 --glmsingle_path="/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/glmsingle_${session}" && \

jupyter nbconvert enhanced_recon_inference.ipynb --to python && \
python enhanced_recon_inference.py --model_name=${model_name} && \

jupyter nbconvert final_evaluations.ipynb --to python && \
python final_evaluations.py --model_name=${model_name} --all_recons_path=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/evals/${model_name}/all_enhancedrecons.pt --data_path=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2 --eval_dir=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/evals/${model_name}
