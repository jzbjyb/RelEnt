#!/bin/bash
#SBATCH --mem=45000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out
set -e

dataset_dir=data/analogy_dataset/by_entail_622_subgraph10_ancestor5_sample100_maxoccperprop10k_haslongtailhasemptysplit
emb_file=${dataset_dir}/emb.txt
subgraph_file=data/subgraph/property_occurrence_all_shuf_top100k.subgraph
#subgraph_file=data/property_occurrence_subtree.subgraph
data_format=pointwise
save_dir=model/large_only_property

python train.py \
    --dataset_dir ${dataset_dir} \
    --dataset_format ${data_format} \
    --subgraph_file ${subgraph_file} \
    --subprop_file data/subprops.txt \
    --emb_file ${emb_file} \
    --patience 40 \
    --num_workers 4 \
    --method ggnn \
    --match concat \
    --batch_size 128 \
    --edge_type only_property \
    --lr 0.001 \
    --neg_ratio 10 \
    --keep_n_per_prop 30:10 \
    --save ${save_dir} \
    --preped
