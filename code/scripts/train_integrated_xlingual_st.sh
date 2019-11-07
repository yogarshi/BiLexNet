#!/bin/sh

#SBATCH --mem=30000
#SBATCH --qos=gpu-short
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=materialgpu02,materialgpu00

module load cuda/8.0.61
module load gcc/5.3.0
module load cudnn/7.5.0

seed=$1
temperature=$2
kdalpha=$3
lr=$4
lang=$5

#echo "Training on $1"
#hostname >> log_${seed}_hi-en-st_${temperature}_${kdalpha}_10eps_kd_pool_atttemp0.1
CUDA_LAUNCH_BLOCKING=1 CUDNN_ROOT=/opt/common/cudnn/cudnn-8.0-5.1  python  train_student-teacher-withmonoen.py \
                                                    --dataset_prefix /fs/clip-xling/CLTE/multilingual-semantic-annotations/to_copy/data/en-${lang}/ \
                                                    --xlingual_path_file /fs/clip-xling/CLTE/multilingual-semantic-annotations/to_copy/data/en-${lang}/all.xlingual.count \
                                                    --mono_path_file_en /fs/clip-xling/CLTE/multilingual-semantic-annotations/to_copy/data/en-${lang}/all.mono.count \
                                                    --mono_path_file_hi /fs/clip-xling/CLTE/multilingual-semantic-annotations/to_copy/data/en-${lang}/train+val+test.mono_hi.paths.count \
                                                    --embeddings_file_en /fs/clip-xling/CLTE/multilingual-semantic-annotations/to_copy/data/en-${lang}/wiki.en.align.vec \
                                                    --embeddings_file_hi /fs/clip-xling/CLTE/multilingual-semantic-annotations/to_copy/data/en-${lang}/wiki.${lang}.align.vec \
                                                    --model_prefix_file /fs/clip-scratch/yogarshi/xlingual-path-exps/modeltmp-st \
							                        --num_hidden_layers 0 -g 1 --lr ${lr} -s ${seed} -t ${temperature} --kdalpha ${kdalpha}  \
							                        >  log_${seed}_${lang}-en_${temperature}_${kdalpha}_5eps

							                        #log_${seed}_hi-en-st_vanillatransfer-embeddingsonly

#SBATCH --gres=gpu:gtxtitanx:3#SBATCH --gres=gpu:gtx1080ti:1
# --nodelist=materialgpu00
#SBATCH --exclude=materialgpu00
#  log_${seed}_hi-en-st_${temperature}_${kdalpha}_5eps_adam_lr${lr}_base
