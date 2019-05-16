#!/bin/bash
#SBATCH --mem=30000
#SBATCH --time=0

data_dir=data/analogy_dataset/by_tree_532_shuf_contain_train_1000_100/

mkdir -p ${data_dir}

python prep_data.py \
    --prop_file data/subprops.txt \
    --prop_dir data/property_occurrence_subtree/ \
    --subgraph_file data/property_occurrence_subtree.subgraph \
    --emb_file ~/tir1/data/wikidata/wikidata_translation_v1.tsv.id.QP \
    --out_dir ${data_dir} \
    --method by_tree \
    --train_dev_test 0.5:0.3:0.2 \
    --max_occ_per_prop 1000 \
    --num_per_prop_pair 100 \
    --contain_train

python train.py \
    --dataset_dir ${data_dir} \
    --subgraph_file data/property_occurrence_subtree.subgraph \
    --subprop_file data/subprops.txt \
    --emb_file ~/tir1/data/wikidata/wikidata_translation_v1.tsv \
    --no_cuda \
    --filter_emb
