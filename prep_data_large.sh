#!/bin/bash
#SBATCH --mem=35000
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out
set -e

data_dir=data/analogy_dataset/by_entail_622_subgraph10_sample100_maxoccperprop10k/
data_format=pointwise
subgraph_file=data/subgraph/property_occurrence_all_shuf_top100k.subgraph

mkdir -p ${data_dir}

python prep_data.py \
    --prop_file data/subprops.txt \
    --prop_dir data/property_occurrence_all_shuf_top100k \
    --subgraph_file $subgraph_file \
    --emb_file ~/tir1/data/wikidata/wikidata_translation_v1.tsv.id.QP \
    --out_dir ${data_dir} \
    --method by_entail \
    --train_dev_test 0.6:0.2:0.2 \
    --max_occ_per_prop 10000 \
    --num_occ_per_subgraph 10 \
    --num_sample 100 \
    --contain_train

python train.py \
    --dataset_dir ${data_dir} \
    --dataset_format ${data_format} \
    --subgraph_file $subgraph_file \
    --subprop_file data/subprops.txt \
    --emb_file ~/tir1/data/wikidata/wikidata_translation_v1.tsv \
    --no_cuda \
    --filter_emb data/emb/test.emb
