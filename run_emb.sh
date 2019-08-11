#!/bin/bash
#SBATCH --mem=20000
#SBATCH --cpus-per-task=8
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out
set -e

subprop_file=data_new/property_occurrence_prop435k_split_dedup/subprops_random

for method in avg mixgau_fast
do
    for emb in ../pytorch_big_graph/emb_new_dedup/transe.txt ../pytorch_big_graph/emb_new_dedup/complex.txt ../pytorch_big_graph/emb_new_dedup/distmult.txt
    do
        for data_dir in "data_new/analogy_dataset/split_middle_overlap_dedup/" "data_new/analogy_dataset/split_middle_overlap_dedup_popu/" "data_new/analogy_dataset/split_middle_overlap_dedup_popu_withtest/"
        do
            echo "======="
            echo ${method} ${emb} ${data_dir}
            echo "======="
            python run_emb.py \
                --dataset_dir ${data_dir} \
                --split test.prop \
                --subprop_file ${subprop_file} \
                --emb_file ${emb} \
                --method ${method} \
                --num_workers 8 \
                --filter_num_poccs 0
        done
    done
done
