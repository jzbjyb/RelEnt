#!/usr/bin/env bash
set -e

# synthesize a dataset suitable for relation hierarchy learning based on the original Wikidata

prop_dir=$1  # a directory holding all the instances of each property.
             # there might be some problems in this directory:
             # (1) it might contain duplcates
             # (2) some properties are useless, like P18 (image)

# get popularity from sling
python prop.py --task get_wikidata_item_popularity_by_sling \
    --inp data/sling_rec/ \
    --out data/sling/mention_popu.tsv

# downsample
python prop.py --task downsample_by_property_and_popularity \
    --inp data/property_occurrence_all/:data/sling/mention_popu.tsv:data/useful_props.tsv \
    --out data_new/property_occurrence_all_ds

# collect ids and inflate
python prop.py --task prop_entities \
    --inp data_new/property_occurrence_all_ds \
    --out data_new/property_eid/property_occurrence_all_ds_notonlytree.eid

python prop.py --task filter_triples \
    --inp data/hiro_wikidata/triples.txt:data_new/property_eid/property_occurrence_all_ds_notonlytree.eid:data/useful_props.tsv \
    --out data_new/hiro_wikidata/triples_prop435k.txt

mkdir -p data_new/property_occurrence_prop435k/
ulimit -n 2048
python prop.py --task prop_occur_all \
    --inp data_new/hiro_wikidata/triples_prop435k.txt \
    --out data_new/property_occurrence_prop435k/

# build ontology
python prop.py --task filter_ontology \
    --inp data/hiro_wikidata/triples.txt \
    --out data/ontology/subclass_instance_triples.txt

python prop.py --task get_partial_order \
    --inp data/ontology/subclass_instance_triples.txt \
    --out data/ontology/node2depth.tsv

# build subgraph
python hiro_code.py \
    --wikitext-dir data/title_id_map \
    --extracted-dir data/hiro_wikidata/ \
    --output-dir data_new/subgraph/ \
    --triple_file data_new/hiro_wikidata/triples_prop435k.txt \
    --eid_file data_new/property_eid/property_occurrence_all_ds_notonlytree.eid

python prop.py --task hiro_to_subgraph \
    --inp data_new/property_eid/property_occurrence_all_ds_notonlytree.eid:data_new/subgraph/hiro_subgraph.jsonl \
    --out data_new/subgraph/property_occurrence_prop435k.subgraph

# split leaves
mkdir data_new/property_occurrence_prop435k_split/
python prop.py --task split_leaf_properties \
    --inp data_new/property_occurrence_prop435k/:data_new/subgraph/property_occurrence_prop435k.subgraph:data/subprops.txt:data/ontology/node2depth.tsv:data/useful_props.tsv \
    --out data_new/property_occurrence_prop435k_split/

# remove duplicates between relations and their ancestors
mkdir data_new/property_occurrence_prop435k_split_dedup/
python prop.py --task remove_dup_between_child_ancestor \
    --inp data_new/property_occurrence_prop435k_split/:data_new/property_occurrence_prop435k_split/subprops \
    --out data_new/property_occurrence_prop435k_split_dedup/
cp data_new/property_occurrence_prop435k_split/subprops data_new/property_occurrence_prop435k_split_dedup/subprops

# merge property instances
python prop.py --task merge_poccs \
    --inp data_new/property_occurrence_prop435k_split_dedup/ \
    --out data_new/split_merge_triples/property_occurrence_prop435k_split_dedup.tsv
shuf data_new/split_merge_triples/property_occurrence_prop435k_split_dedup.tsv > \
    data_new/split_merge_triples/property_occurrence_prop435k_split_dedup_shuf.tsv

# generate new subgraphs
python hiro_code.py \
    --wikitext-dir data/title_id_map \
    --extracted-dir data/hiro_wikidata/ \
    --output-dir data_new/subgraph/ \
    --triple_file data_new/split_merge_triples/property_occurrence_prop435k_split_dedup.tsv \
    --eid_file data_new/property_eid/property_occurrence_all_ds_notonlytree.eid

python prop.py --task hiro_to_subgraph \
    --inp data_new/property_eid/property_occurrence_all_ds_notonlytree.eid:data_new/subgraph/hiro_subgraph.jsonl \
    --out data_new/subgraph/property_occurrence_prop435k_split_dedup.subgraph

# train KGE models and put it to ${emb}

# replace parents
python prop.py --task replace_by_hard_split \
    --inp data_new/property_occurrence_prop435k_split_dedup/subprops:${emb} \
    --out data_new/property_occurrence_prop435k_split_dedup/subprops_random
