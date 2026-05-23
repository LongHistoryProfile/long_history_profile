set -x

MODEL_PATH=   # models/Qwen3-8B # set your model path here

WORK_DIR=long_history_profile # set your work directory here, absolute path recommended
cd $WORK_DIR

BASE_LOG_PATH="/logs/training" #set your log path here
START_IDX=0
EPOCH=1

TIME=$(date +"%Y%m%d_%H%M%S")
RUN_NAME=${TIME}


LOG_PATH="$BASE_LOG_PATH/$RUN_NAME"
SAVE_DIR="$LOG_PATH/trainer"
VALIDATION_PATH="$LOG_PATH/validation"
ROLLOUT_DATA_DIR="$LOG_PATH/rollout_data"
LOG_FILE="$LOG_PATH/log.txt"

mkdir -p "$SAVE_DIR" "$VALIDATION_PATH" "$ROLLOUT_DATA_DIR"

cd verl

export WANDB_API_KEY=   # your wandb api key here
export WANDB_ENTITY=   # your wandb entity here

TRAIN_PATH=   # set your train path here 

# ============ util ============
get_max_global_step () {
    local dir=$1

    if [ ! -d "$dir" ]; then
        echo "None"
        return
    fi

    local step
    step=$(ls "$dir" 2>/dev/null \
        | sed -n 's/^\([0-9]\+\)_.*/\1/p' \
        | sort -n \
        | tail -1)

    if [ -z "$step" ]; then
        echo "None"
    else
        echo "$step"
    fi
}


# ============ training loop ============
while true; do
    GLOBAL_STEPS=$(get_max_global_step "$VALIDATION_PATH")

    echo "========================================"
    echo "START_IDX     = $START_IDX"
    echo "EPOCH         = $EPOCH"
    echo "GLOBAL_STEPS  = $GLOBAL_STEPS"
    echo "========================================"

    HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.custom_cls.path=$WORK_DIR/verl/verl/utils/dataset/profile_dataset_v2.py \
        data.custom_cls.name=MultiTurnUserHistoryDataset \
        +data.train_path=$TRAIN_PATH \
        +data.test_path=$TRAIN_PATH \
        +data.start_idx=$START_IDX \
        +data.episode_step_size=10 \
        +data.prev_validation_data_dir="$VALIDATION_PATH" \
        +data.prev_global_steps=$GLOBAL_STEPS \
        +data.load_dataloader_state=False \
        data.train_batch_size=32 \
        data.val_batch_size=96 \
        data.max_prompt_length=8192 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        data.return_raw_chat=True \
        data.return_full_prompt=True \
        custom_reward_function.path=$WORK_DIR/verl/verl/utils/reward_score/lstm_profile.py \
        custom_reward_function.name=profile_controller_compute_reward \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        +trainer.trainer_cls=verl.trainer.ppo.my_ray_trainer_v2.LSTMRayPPOTrainer \
        trainer.default_local_dir=$SAVE_DIR \
        trainer.validation_data_dir=$VALIDATION_PATH \
        trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='LONG_HISTORY_PROFILE' \
        trainer.experiment_name="'${RUN_NAME}_idx_${START_IDX}'" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=4000000000 \
        trainer.total_epochs=$EPOCH \
        trainer.val_before_train=False \
        2>&1 | tee -a "$LOG_FILE"

    # ===== update for next round =====
    START_IDX=$((START_IDX + 10))
    EPOCH=$((EPOCH + 1))
done
