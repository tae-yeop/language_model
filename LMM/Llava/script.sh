#/bin/bash -l

#SBATCH --job-name=vlm-finetune
#SBATCH --time=999:000
#SBATCH -p 80g
#SBATCH --gres=gpu:8
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH --comments="VLM 파인튜닝"

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/nano_vlm2.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MHY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/LLM/Llava'

srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    python trl_finetuning.py
