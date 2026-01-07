#!/bin/bash
#SBATCH --job-name=sub-005_ses-01_finalmask_MST
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1              
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --gpus-per-task=1       # Set to equal gres=gpu:#!
#SBATCH --cpus-per-task=40      # 40 / 80 / 176 distributed across node
#SBATCH --time=01:30:00         # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%A_%a.err     # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%A_%a.out
#SBATCH --no-requeue
#SBATCH --array=0               # 0 or 0-9
#SBATCH --mail-type=END
#SBATCH --mail-user=rsiyer@princeton.edu

mkdir -p slurms   # create directory if it doesn't exist

echo "My SLURM_ARRAY_JOB_ID is ${SLURM_ARRAY_JOB_ID}"
echo "My SLURM_ARRAY_TASK_ID is ${SLURM_ARRAY_TASK_ID}"
echo "Executing on the machine: $(hostname)"

module purge
module load anaconda3/2023.3
module load fsl/6.0.6.2
conda activate rt_mindEye2

# verify these variables before submitting
# ---
sub="sub-005"
train_session="ses-01"                  # session to train on
test_session="ses-03"                   # session to test on (can be the same as train_session)
split="repeats_3"                       # MST, orig, unique, repeats_3
task="C"
suffix="_avgrepeats_finalmask_150epochs"          # appended to model_name

run_training="True"                    
run_recon_inference="True"
run_enhanced_recon="False" 
run_final_evals="True"

main_script="main-multisession-3tasks"
# --- 

# conditionals based on config provided above
# --- 
if [ "${run_training}" = "False" ] && [ "${run_recon_inference}" = "False" ] && [ "${run_enhanced_recon}" = "False" ] && [ "${run_final_evals}" = "False" ]; then
  echo "Error: At least one of run_training, run_recon_inference, run_enhanced_recon, or run_final_evals must be True"
  exit 1
fi

train_glmsingle_path="/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/glmsingle_${sub}_${train_session}_task-${task}"
test_glmsingle_path="/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/glmsingle_${sub}_${test_session}_task-${task}"
[ ! -d "${train_glmsingle_path}" ] && echo "ERROR: Train path not found: ${train_glmsingle_path}" && exit 1
[ ! -d "${test_glmsingle_path}" ] && echo "ERROR: Test path not found: ${test_glmsingle_path}" && exit 1

model_name="${sub}_${train_session}_task-${task}_bs24_MST_rishab_${split}split_${SLURM_ARRAY_TASK_ID}${suffix}"  # model should always be based on train_session
eval_dir_template="${model_name}"

if [ "${train_session}" != "${test_session}" ]; then
  mask_from_other_session=${train_session}  # mask should be from the original session that was trained on
  eval_dir_template="${model_name}_test-${test_session}"
else
  mask_from_other_session=""
fi

eval_dir="/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/evals/${eval_dir_template}"
# ---

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
echo model_name=${model_name}

export sub=${sub}
export task=${task}
export split=${split}
export eval_dir=${eval_dir}
export mask_from_other_session=${mask_from_other_session:-""}  # empty string default if not set
export WANDB_MODE="offline"

echo "Configuration:"
echo "  Subject: ${sub}"
echo "  Train session: ${train_session}"
echo "  Test session: ${test_session}"
echo "  Run training: ${run_training}"
echo "  Run recon inference: ${run_recon_inference}"
echo "  Run enhanced recon: ${run_enhanced_recon}"
echo "  Run final evaluations: ${run_final_evals}"
echo "  Model name: ${model_name}"
echo "  Train data: ${train_glmsingle_path}"
echo "  Test data: ${test_glmsingle_path}"
echo "  Mask from: ${mask_from_other_session:-same session}"
echo "  Eval dir: ${eval_dir}"

# fine-tuning on new data. if train_on_different_session is provided, don't do training because we only want evaluation
if [ "${run_training}" = "True" ]; then
    export session=${train_session}
    export glmsingle_path=${train_glmsingle_path}
  
    jupyter nbconvert "${main_script}.ipynb" --to python && \
    accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT "${main_script}.py" --data_path=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2 --model_name=${model_name} --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --no-use_image_aug --blur_scale=.5 --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/train_logs/multisubject_subj01_1024hid_nolow_300ep --seed="${SLURM_ARRAY_TASK_ID}" && \
    echo "Training completed successfully"
else
    echo "Skipping training, using existing checkpoint for ${model_name}"
fi

export session=${test_session}

if [ "${run_recon_inference}" = "True" ]; then
    jupyter nbconvert recon_inference-multisession.ipynb --to python && \
    python recon_inference-multisession.py --model_name=${model_name} --subj=1 --no-blurry_recon --use_prior --hidden_dim=1024 --n_blocks=4 --glmsingle_path="${test_glmsingle_path}" --eval_dir=${eval_dir} && \
    echo "recon_inference completed successfully"
else
    echo "Skipping recon inference for ${model_name}"
fi

if [ "${run_enhanced_recon}" = "True" ]; then
    jupyter nbconvert enhanced_recon_inference.ipynb --to python && \
    python enhanced_recon_inference.py --model_name=${model_name} --all_recons_path=${eval_dir}/${model_name}_all_recons.pt && \
    echo "enhanced_recon_inference completed successfully"
else
    echo "Skipping enhanced recon inference for ${model_name}"
fi

if [ "${run_final_evals}" = "True" ]; then
    jupyter nbconvert final_evaluations.ipynb --to python && \
    python final_evaluations.py --model_name=${model_name} --all_recons_path=${eval_dir}/${model_name}_all_recons.pt --data_path=/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2 --eval_dir=${eval_dir} && \
    echo "final_evaluations completed successfully"
else
    echo "Skipping final evaluations for ${model_name}"
fi

echo "Remember to sync wandb logs with online node!"
