#!/usr/bin/env bash
set -e

ulimit -n 10000


for fid in 0 1 2 3 4 5 6 7 8 9; do
    dir_fid=data_new/property_occurrence_prop435k_split_dedup_sling/${fid}_dep_2
    #echo remove ${dir_fid}
    #rm -rf ${dir_fid}
    mkdir -p ${dir_fid}
done

for fid in 0 1 2 3 4 5 6 7 8 9; do
    python prop.py --task get_entity_occ_dep_by_sling \
        --inp data_new/property_occurrence_prop435k_split_dedup_sling/${fid} \
        --out data_new/property_occurrence_prop435k_split_dedup_sling/${fid}_dep_2
done
