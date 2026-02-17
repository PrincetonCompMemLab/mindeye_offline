#!/bin/bash
#SBATCH --job-name=bixby_mindeye
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1              
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --gpus-per-task=1       # Set to equal gres=gpu:#!
#SBATCH --cpus-per-task=40      # 40 / 80 / 176 distributed across node
#SBATCH --time=03:00:00         # total run time limit (HH:MM:SS)
#SBATCH -e bixby_data/slurms/%A_%a.err     # first create a "slurms" folder in current directory to store logs
#SBATCH -o bixby_data/slurms/%A_%a.out
#SBATCH --no-requeue
#SBATCH --array=01               
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wg7536@princeton.edu


mkdir -p bixby_data/slurms   # create directory if it doesn't exist

echo "My SLURM_ARRAY_JOB_ID is ${SLURM_ARRAY_JOB_ID}"
echo "My SLURM_ARRAY_TASK_ID is ${SLURM_ARRAY_TASK_ID}"
echo "Executing on the machine: $(hostname)"

module purge
module load anaconda3/2023.3
module load fsl/6.0.6.2
conda activate rt_mindEye2

# Set subject ID based on array index
printf -v sub "%02d" $SLURM_ARRAY_TASK_ID

sub="sub-${sub}"

echo "Subject ID: ${sub}"

suffix="_avgrepeats_unionmask_150epochs"          # appended to model_name

run_training="True"                   
run_final_evals="False"

main_script="bixby-main-multisession_afni"

model_name="${sub}_3-session_task-mindeye_${SLURM_ARRAY_TASK_ID}${suffix}"  # model should always be based on train_session
eval_dir_template="${model_name}"
eval_dir="/scratch/gpfs/KNORMAN/wanjia/mindeye_testing/real_time_mindEye2/bixby_data/evals/${eval_dir_template}"
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
export WANDB_MODE="offline"

echo "Configuration:"
echo "  Subject: ${sub}"
echo "  Run training: ${run_training}"
echo "  Run final evaluations: ${run_final_evals}"
echo "  Model name: ${model_name}"
echo "  Eval dir: ${eval_dir}"

# fine-tuning on new data. if train_on_different_session is provided, don't do training because we only want evaluation
if [ "${run_training}" = "True" ]; then

    jupyter nbconvert "${main_script}.ipynb" --to python && \
    accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT "${main_script}.py" --model_name=${model_name} --subj=${sub} --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --ckpt_interval=999 --ckpt_saving --seed="${SLURM_ARRAY_TASK_ID}" && \
    echo "Training completed successfully"
else
    echo "Skipping training, using existing checkpoint for ${model_name}"
fi