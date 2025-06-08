#!/bin/bash -l

#SBATCH --job-name=chemberta
#SBATCH --time=999:000
#SBATCH --partition=80g
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --comment="ChemBERTa 테스트"
#SBATCH --output=chemberta_%j.out

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/laidd5.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/MM/LLM/ChemBERTa'

srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    python finetune.py --nnodes $SLURM_NNODES --ngpus $SLURM_NTASKS_PER_NODE