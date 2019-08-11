#!/bin/bash
#SBATCH --mem=20000
#SBATCH --cpus-per-task=8
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out
set -e

dataset_dir=data_new/analogy_dataset/split_middle_dedup_nocommonchild_by_entail_nway_subgraph10_sample5_parent_occ_popu/
subprop_file=data_new/property_occurrence_prop435k_split_dedup/subprops_random
emb_file=../pytorch_big_graph/emb_new_dedup/transe.txt
#word_emb_file='data/emb/glove.6B.50d.txt'
#word_emb_size=50
#suffix='.tbow'
word_emb_file='../pytorch_big_graph/emb_new_dedup/transe.txt'
word_emb_size=200
suffix='.kg_tbow'
word_emb_file2='data/emb/glove.6B.50d.txt'
word_emb_size2=50
suffix2='.tbow'

for seed in 0 1 2 3 4 2019 1000 256 77 9541
do
    for use_tbow in 0
    do
        echo "======="
        echo ${seed} ${use_tbow}
        echo "======="
        python run_emb_learn.py \
            --dataset_dir ${dataset_dir} \
            --subprop_file ${subprop_file} \
            --emb_file ${emb_file} \
            --word_emb_file ${word_emb_file} \
            --word_emb_size ${word_emb_size} \
            --use_tbow ${use_tbow} \
            --suffix ${suffix} \
            --word_emb_file2 ${word_emb_file2} \
            --word_emb_size2 ${word_emb_size2} \
            --use_tbow2 ${use_tbow} \
            --suffix2 ${suffix2} \
            --lr 0.001 \
            --epoch 500 \
            --early_stop 100 \
            --seed ${seed} \
            --only_one_sample_per_prop \
            --filter_labels \
            --use_gnn
    done
done
