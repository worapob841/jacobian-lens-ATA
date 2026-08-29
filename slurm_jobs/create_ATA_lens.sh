#!/bin/bash

#SBATCH --job-name=mllm         # Job name
#SBATCH --qos=uninet-limit64cores500gb
#SBATCH --partition=UniNet      # Specify the partition
#SBATCH --gres=gpu:4           # Request 4 GPUs
#SBATCH --cpus-per-task=16           # Request 64 CPU cores
#SBATCH --mem=64G                     # Request 64 GB of memory
#SBATCH --time=3-00:00:00             # Max runtime (3 days)
#SBATCH --output=sbatch_logs/%j_ata-lens/ata-lens_output_%j.log        # Standard output log

# Load Singularity module if needed
module load singularity
mkdir -p "/g/home/orachat.c/torch_tmpdir"

export TMPDIR=/g/home/orachat.c/torch_tmpdir
export PYTHONPATH=/g/home/orachat.c/project/MLLM/TokenPacker/llava:$PYTHONPATH
export TOKENPACKER_REPO=/g/home/orachat.c/project/MLLM/TokenPacker
# Use Singularity to execute the environment within the .sif file
singularity exec --nv   \
        /g/home/orachat.c/project/MLLM/TokenPacker/research.sif \
            torchrun --nproc_per_node=4 fit_multimodal_distributed.py \
        --model_path ../TokenPacker/checkpoints/llava-cross_attn_adaptive-it-spatialscorrer-thres4060-bound-3060-alpha-001-multilev-dubconv-llava_v1_5_mix665k-en-h100-08092026 \
        --question_file ../TokenPacker/playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl \
        --image_folder ../TokenPacker/playground/data/eval/vqav2/test2015 \
        --n_samples 200 \
        --dim_batch 16 \
        --max_seq_len 256 \
        --output_lens_path out/llava-cross_attn_adaptive-it-spatialscorrer-thres4060-bound-3060-alpha-001-multilev-dubconv-llava_v1_5_mix665k-en-h100-08092026/llava-cross_attn_adaptive-it-spatialscorrer-thres4060-bound-3060-alpha-001-multilev-dubconv-llava_v1_5_mix665k-en-h100-08092026_multimodal_vqav2_lens.pt