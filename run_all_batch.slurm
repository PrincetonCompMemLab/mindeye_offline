#!/bin/bash
#SBATCH --job-name=sub-005_ses-01-02
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1              
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --gpus-per-task=1       # Set to equal gres=gpu:#!
#SBATCH --cpus-per-task=40      # 40 / 80 / 176 distributed across node
#SBATCH --time=02:00:00         # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%A_%a.err     # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%A_%a.out
#SBATCH --no-requeue
#SBATCH --array=0               # 0 or 0-9
#SBATCH --mail-type=END
#SBATCH --mail-user=rsiyer@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is ${SLURM_ARRAY_JOB_ID}"
echo "My SLURM_ARRAY_TASK_ID is ${SLURM_ARRAY_TASK_ID}"
echo "Executing on the machine: $(hostname)"

module purge
module load anaconda3/2023.3
module load fsl/6.0.6.2
conda activate rt_mindEye2
# source /scratch/gpfs/ri4541/MindEyeV2/src/fmri/bin/activate

# verify these variables before submitting
# ---
sub="sub-005"
session="all"
session_label='ses-01-02'
split=MST  # MST train/test split, alternative would be train on non-repeats and test on images that repeat (split=orig) 
task=C
# model_name="${sub}_multi_task-${task}_bs24_MST_rishab_${split}split"
model_name="${sub}_${session}_task-${task}_bs24_MST_rishab_${split}split_${SLURM_ARRAY_TASK_ID}"
main_script="main-multisession-3tasks"
# glmsingle_path="/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/glmsingle-multi"
glmsingle_path="/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/glmsingle_${sub}_${session_label}_task-${task}"
# glmsingle_path="/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/glmsingle_${sub}_${session}"
# --- 

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=24
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

echo model_name=${model_name}

eval_dir="/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/evals/${model_name}"
export sub=${sub}
export session=${session}
export task=${task}
export glmsingle_path=${glmsingle_path}
export eval_dir=${eval_dir}
export WANDB_MODE="offline"

# singlesubject finetuning
jupyter nbconvert "${main_script}.ipynb" --to python && \
accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT "${main_script}.py" --data_path=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2 --model_name=${model_name} --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/train_logs/multisubject_subj01_1024hid_nolow_300ep --seed="${SLURM_ARRAY_TASK_ID}" && \

# jupyter nbconvert recon_inference-multisession.ipynb --to python && \
# python recon_inference-multisession.py --model_name=${model_name} --subj=1 --no-blurry_recon --use_prior --hidden_dim=1024 --n_blocks=4 --glmsingle_path="${glmsingle_path}" && \

# #jupyter nbconvert recon_inference_orig.ipynb --to python && \
# #python recon_inference_orig.py --model_name=${model_name} --subj=1 --no-blurry_recon --use_prior --hidden_dim=1024 --n_blocks=4 && \ 

# jupyter nbconvert enhanced_recon_inference.ipynb --to python && \
# python enhanced_recon_inference.py --model_name=${model_name} --all_recons_path=${eval_dir}/${model_name}_all_recons.pt && \

# #jupyter nbconvert enhanced_recon_inference_orig.ipynb --to python && \
# #python enhanced_recon_inference_orig.py --model_name=${model_name} && \

# jupyter nbconvert final_evaluations.ipynb --to python && \
# python final_evaluations.py --model_name=${model_name} --all_recons_path=${eval_dir}/all_enhancedrecons.pt --data_path=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2 --eval_dir=${eval_dir} && \

#jupyter nbconvert final_evaluations_orig.ipynb --to python && \
#python final_evaluations_orig.py --model_name=${model_name} --all_recons_path=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/evals/${model_name}/${model_name}_all_recons.pt --data_path=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2 --eval_dir=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/evals/${model_name}

echo "Remember to sync wandb logs with online node!"
