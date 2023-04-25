#!/bin/bash
#SBATCH --job-name=RESNET_CLIP_s1000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:4
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=48G
#SBATCH --partition=gpu4_medium
#SBATCH --output=./RESNET_CLIP_s1000.out
#SBATCH --error=./RESNET_CLIP_s1000.err

# activate conda env
source activate deeplearning_general

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python ./main_pretrain.py --batch_size 64 --gpus 4 --num_nodes 1 --max_epochs 50 --lr_img_backbone 1e-4 --lr_text_backbone 1e-4 --img_backbone "resnet2d_50" --max_length 100 --img_embedding_dim 2048 --weight_decay 0.01 --optimizer "adamw" --method "CLIP" --save_dir "model_saved" --pretrained --seed 1000 --num_workers 16 --temperature_mm 0.07 --per_warmup_steps 0.0

