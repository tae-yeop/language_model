#!/bin/bash -l

#SBATCH --job-name=llm
#SBATCH --time=999:000
#SBATCH --partition=80g
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --comment="qwen3"
#SBATCH --output=qwen_ft_%j.out

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/llm_27_v3.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/LLM/5_sft/qwen3'

export HF_TOKEN=''
export WANDB_KEY=''

srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    torchrun --nproc_per_node=4 train.py --hf_token $HF_TOKEN \
                                        --wandb_key $WANDB_KEY \
                                        --do_train \
                                        --logging_strategy steps \
                                        --logging_steps 1 \
                                        --per_device_train_batch_size 1 \
                                        --gradient_accumulation_steps 16


# 12, 8는 OOM
# https://www.datacamp.com/tutorial/fine-tuning-qwen3
# 엄청 큰 모델도 LoRA를 써서 학습