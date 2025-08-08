#!/bin/bash
#SBATCH --job-name=ses-01
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1              
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --gpus-per-task=1       # Set to equal gres=gpu:#!
#SBATCH --cpus-per-task=40      # 40 / 80 / 176 distributed across node
#SBATCH --time=01:20:00         # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --no-requeue
#SBATCH --array=0-19
#SBATCH --mail-type=END
#SBATCH --mail-user=rsiyer@princeton.edu

module purge
module load anaconda3/2023.3
# conda activate rt_mindEye2
source /scratch/gpfs/ri4541/MindEyeV2/src/fmri/bin/activate

# The following line converts your jupyter notebook into a python script runnable with Slurm
jupyter nbconvert main-multisession.ipynb --to python

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=24
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

# singlesubject finetuning
model_name="sub-001_ses-01_bs24_MST_rishab_MSTsplit_orig"
echo model_name=${model_name}
accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT orig_main.py --data_path=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2 --model_name=${model_name} --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-5 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --multisubject_ckpt=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/train_logs/multisubject_subj01_1024hid_nolow_300ep
