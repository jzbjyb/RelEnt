#!/bin/bash
#SBATCH --mem=20000
#SBATCH --cpus-per-task=8
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out
set -e

for method in avg mixgau_fast
do
    for emb in ../pytorch_big_graph/emb/transe.txt ../pytorch_big_graph/emb/complex.txt ../pytorch_big_graph/emb/distmult.txt
    do
        for data_dir in "data/analogy_dataset/split_middle_overlap/" "data/analogy_dataset/split_middle_overlap_popu/" "data/analogy_dataset/split_middle_overlap_popu_withtest/"
        do
            echo "======="
            echo ${method} ${emb} ${data_dir}
            echo "======="
            python run_emb.py \
                --dataset_dir ${data_dir} \
                --split test.prop \
                --subprop_file data/property_occurrence_prop580k_split/subprops_hard \
                --emb_file ${emb} \
                --method ${method} \
                --num_workers 8 \
                --filter_num_poccs 0
        done
    done
done
