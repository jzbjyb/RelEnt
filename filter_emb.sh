#!/bin/bash
#SBATCH --mem=10000
#SBATCH --time=0

python train.py --dataset_dir data/analogy_dataset/within_tree_532/ \
    --subgraph_file data/property_occurrence_subtree.subgraph --subprop_file data/subprops.txt \
    --emb_file ~/tir1/data/wikidata/wikidata_translation_v1.tsv --no_cuda --filter_emb