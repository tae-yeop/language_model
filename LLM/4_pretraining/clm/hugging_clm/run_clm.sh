#!/bin/bash -l

#SBATCH --job-name=llm
#SBATCH --time=999:000
#SBATCH --partition=80g
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --comment="CLM"
#SBATCH --output=clm_%j.out

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/llm_v3.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/LLM/4_pretraining/clm/hugging_clm'

srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    torchrun --nproc_per_node=8 run_clm.py --model_name_or_path openai-community/gpt2 \
                    --dataset_name wikitext \
                    --dataset_config_name wikitext-2-raw-v1 \
                    --per_device_train_batch_size 8 \
                    --per_device_eval_batch_size 8 \
                    --do_train \
                    --do_eval \
                    --logging_strategy steps \
                    --logging_steps 1 \
                    --eval_strategy steps \
                    --eval_steps 50 \
                    --save_strategy steps \
                    --save_steps 1000 \
                    --output_dir /purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/LLM/4_pretraining/clm/hugging_clm/out \
                    --report_to wandb \
                    --wandb_key local-73177de041f41c769eb8cbdccb982a9a5406fab7 \
                    --wandb_run_name test3

# eval_strategy ← 중간 평가 켜기
# eval_steps ← 주기
# ave_strategy ← 같은 주기로 체크포인트 저장