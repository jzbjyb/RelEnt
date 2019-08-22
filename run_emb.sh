#!/bin/bash
#SBATCH --mem=20000
#SBATCH --cpus-per-task=8
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out
set -e

subprop_file=data_new/analogy_dataset/split_middle_dedup_nocommonchild_by_entail_nway_subgraph100_sample5_parent_occ_popu/subprops
data_dir_main="data_new/analogy_dataset/split_middle_overlap_dedup_nocommonchild/"

top=10

for method in avg kde
do
    for emb in ../pytorch_big_graph/emb_new_dedup_shuf_default/transe.txt ../pytorch_big_graph/emb_new_dedup_shuf_default/transe_cos.txt ../pytorch_big_graph/emb_new_dedup_shuf_default/complex.txt ../pytorch_big_graph/emb_new_dedup_shuf_default/distmult.txt
    do
        for data_dir in "data_new/analogy_dataset/split_middle_overlap_dedup_nocommonchild/" "data_new/analogy_dataset/split_middle_overlap_dedup_nocommonchild_popu/" "data_new/analogy_dataset/split_middle_overlap_dedup_nocommonchild_popu_withtest/"
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
                --filter_num_poccs 0 \
                --top ${top} \
                --seed 2019 \
                --sigma 0.1 \
                --use_norm
        done

        if [[ $1 == 'prop' ]]; then
            echo "======="
            echo only_prop ${emb}
            echo "======="
            python run_emb.py \
                --dataset_dir ${data_dir_main} \
                --split test.prop \
                --subprop_file ${subprop_file} \
                --emb_file ${emb} \
                --method avg \
                --num_workers 8 \
                --filter_num_poccs 0 \
                --top ${top} \
                --seed 2019 \
                --sigma 0.1 \
                --use_norm \
                --only_prop_emb
        fi

    done
done
