#!/bin/bash
#SBATCH --mem=35000
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out
set -e

data_dir=data_new/analogy_dataset/split_middle_dedup_nocommonchild_by_entail_nway_subgraph100_sample5_parent_occ_popu_label/
subprop_file=data_new/property_occurrence_prop435k_split_dedup/subprops_random
emb_file=data/emb/glove.6B.50d.txt
pid2snippet_file=data_new/textual/wikipedia_sling/pid2snippet.pkl
dep_dir=data_new/property_occurrence_prop435k_split_dedup_sling

for out in bow sent all
do
    python prep_text_data.py \
        --data_dir ${data_dir} \
        --subprop_file ${subprop_file} \
        --emb_file ${emb_file} \
        --pid2snippet_file ${pid2snippet_file} \
        --out ${out} \
        --suffix ".mid_${out}"

    python prep_text_data.py \
        --data_dir ${data_dir} \
        --subprop_file ${subprop_file} \
        --emb_file ${emb_file} \
        --dep_dir ${dep_dir} \
        --out ${out} \
        --suffix ".dep_${out}"
done
