#!/bin/bash
#SBATCH --mem=45000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out
set -e

#dataset_dir=data/analogy_dataset/by_entail_n_way_622_subgraph10_sample1000
#dataset_dir=data/analogy_dataset/by_entail_n_way_622_subgraph10_sample100
dataset_dir=data/analogy_dataset/by_entail_622_subgraph10_ancestor5_sample100_maxoccperprop10k_population
subgraph_file=data/subgraph/property_occurrence_all_shuf_top100k.subgraph
#subgraph_file=data/property_occurrence_subtree.subgraph
data_format=pointwise

python train.py \
    --dataset_dir ${dataset_dir} \
    --dataset_format ${data_format} \
    --subgraph_file ${subgraph_file} \
    --subprop_file data/subprops.txt \
    --emb_file ${dataset_dir}/emb.txt.gz \
    --patience 40 \
    --save model/ggnn.bin \
    --preped \
    --num_workers 4 \
    --method ggnn \
    --batch_size 32
