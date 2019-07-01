#!/bin/bash
#SBATCH --mem=45000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out
set -e

dataset_dir=data/analogy_dataset/split_middle_by_entail_nway_subgraph10_sample5
emb_file=../pytorch_big_graph/emb/transe.txt
subgraph_file=data/subgraph/property_occurrence_prop580k_split.subgraph
subprop_file=data/property_occurrence_prop580k_split/subprops_hard
data_format=nway
save_dir=model/test

python train.py \
    --dataset_dir ${dataset_dir} \
    --dataset_format ${data_format} \
    --subgraph_file ${subgraph_file} \
    --subprop_file ${subprop_file} \
    --emb_file ${emb_file} \
    --patience 40 \
    --num_workers 4 \
    --method ggnn \
    --match concat \
    --batch_size 128 \
    --edge_type bow \
    --lr 0.001 \
    --neg_ratio 10 \
    --keep_n_per_prop 30:10 \
    --save ${save_dir} \
    --preped
