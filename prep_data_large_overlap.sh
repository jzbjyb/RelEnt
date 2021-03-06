#!/bin/bash
#SBATCH --mem=35000
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out
set -e

data_dir=data_new/analogy_dataset/split_middle_overlap_dedup_nocommonchild_popu
load_split=data_new/analogy_dataset/split_middle_overlap_dedup_nocommonchild
data_format=n_way
subgraph_file=data_new/subgraph/property_occurrence_prop435k_split_dedup.subgraph
prop_dir=data_new/property_occurrence_prop435k_split_dedup
prop_file=data_new/property_occurrence_prop435k_split_dedup/subprops_random
method=by_entail-overlap
emb_file=../pytorch_big_graph/emb_new_dedup/transe.txt
#entityid2name_file=data_new/split_merge_triples/property_occurrence_prop580k_split_entityid2name.pkl

mkdir -p ${data_dir}

python prep_data.py \
    --prop_file ${prop_file} \
    --prop_dir ${prop_dir} \
    --subgraph_file ${subgraph_file} \
    --emb_file ${emb_file} \
    --out_dir ${data_dir} \
    --method ${method} \
    --train_dev_test 0.6:0.2:0.2 \
    --max_occ_per_prop 10000 \
    --min_occ_per_prop 10 \
    --num_occ_per_subgraph 10 \
    --num_sample 5 \
    --contain_train \
    --allow_empty_split \
    --remove_common_child \
    --load_split ${load_split} \
    --property_population \
    --filter_test

: '
python train.py \
    --dataset_dir ${data_dir} \
    --dataset_format ${data_format} \
    --subgraph_file ${subgraph_file} \
    --subprop_file data/subprops.txt \
    --emb_file data/emb/wikidata_translation_v1.tsv \
    --no_cuda \
    --filter_emb data/emb/test.emb \
    --method ggnn \
    --edge_type one
'
