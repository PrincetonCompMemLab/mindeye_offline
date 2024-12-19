#!/bin/bash
echo "Executing on the machine: $(hostname)"

module purge
source ~/rt_mindeye/bin/activate

# verify these variables before submitting
# ---
sub=sub-001
session=ses-01
split=MST  # MST train/test split, alternative would be train on non-repeats and test on images that repeat (split=orig)
model_name="${sub}_${session}_bs24_MST_rishab_${split}split"
main_script='main-multisession'
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

# singlesubject finetuning
jupyter nbconvert "${main_script}.ipynb" --to python && \
accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT "${main_script}.py" --data_path=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2 --model_name=${model_name} --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/train_logs/multisubject_subj01_1024hid_nolow_300ep --seed="${SLURM_ARRAY_TASK_ID}" && \

jupyter nbconvert recon_inference-multisession.ipynb --to python && \
python recon_inference-multisession.py --model_name=${model_name} --subj=1 --no-blurry_recon --use_prior --hidden_dim=1024 --n_blocks=4 --glmsingle_path="/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/glmsingle_${session}_paul" && \

jupyter nbconvert enhanced_recon_inference.ipynb --to python && \
python enhanced_recon_inference.py --model_name=${model_name} --all_recons_path=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/evals/${model_name}/${model_name}_all_recons.pt && \

jupyter nbconvert final_evaluations.ipynb --to python && \
python final_evaluations.py --model_name=${model_name} --all_recons_path=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/evals/${model_name}/all_enhancedrecons.pt --data_path=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2 --eval_dir=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/evals/${model_name}
