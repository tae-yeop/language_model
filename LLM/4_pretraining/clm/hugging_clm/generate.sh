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
    python generate.py