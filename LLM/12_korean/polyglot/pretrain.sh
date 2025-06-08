#!/bin/bash -l

#SBATCH --job-name=polygpt
#SBATCH --time=999:000
#SBATCH -p 80g
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --comment="한국어 GPT 테스트"
#SBATCH --output=polyglot_%j.out

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/nano_vlm5.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/MM/LLM/polyglot'

srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    python pretrain.py --nnodes $SLURM_NNODES --ngpus $SLURM_NTASKS_PER_NODE \