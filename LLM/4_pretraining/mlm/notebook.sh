#!/bin/bash
#SBATCH --job-name=notebook
#SBATCH --time=999:000
#SBATCH --partition=80g
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --output=notebook.out
#SBATCH --comment="RAG 테스트"

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/laidd5.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/MM/LLM/language_modeling'

# --NotebookApp.token='' --NotebookApp.allow_remote_access=True 
srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    jupyter notebook --ip 0.0.0.0 --NotebookApp.allow_origin='*' --NotebookApp.trust_xheaders=True --NotebookApp.disable_check_xsrf=True