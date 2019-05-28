#!/bin/bash
#SBATCH --mem=30000
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out
set -e

dataset_dir=data/analogy_dataset/by_entail_n_way_622_subgraph10_sample1000
#dataset_dir=data/analogy_dataset/by_entail_n_way_622_subgraph10_sample100
subgraph_file=data/subgraph/property_occurrence_all_shuf_top100k.subgraph
#subgraph_file=data/property_occurrence_subtree.subgraph

python train.py \
    --dataset_dir $dataset_dir \
    --dataset_format nway \
    --subgraph_file $subgraph_file \
    --subprop_file data/subprops.txt \
    --emb_file ${dataset_dir}/emb \
    --patience 40 \
    --preped
    #--preped
    #--prep_data
