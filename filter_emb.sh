#!/bin/bash
#SBATCH --mem=30000
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out

data_dir=data/analogy_dataset/by_tree_532_shuf_contain_train_1000_100/

python train.py \
    --dataset_dir ${data_dir} \
    --subgraph_file data/property_occurrence_subtree_hop2.subgraph \
    --subprop_file data/subprops.txt \
    --emb_file ~/tir1/data/wikidata/wikidata_translation_v1.tsv \
    --no_cuda \
    --filter_emb
