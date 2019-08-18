#!/usr/bin/env bash
set -e

ulimit -n 10000

for fid in 0 1 2 3 4 5 6 7 8 9; do
    dir_fid=data_new/property_occurrence_prop435k_split_dedup_sling/${fid}
    #echo remove ${dir_fid}
    #rm -rf ${dir_fid}
    mkdir -p ${dir_fid}
done

for fid in 0 1 2 3 4 5 6 7 8 9; do
    python prop.py --task get_entity_occ_by_sling \
        --inp data_new/property_occurrence_prop435k_split_dedup/subprops_random:data_new/property_occurrence_prop435k_split_dedup/:data/sling_rec/:documents-0000${fid}-of-00010.rec \
        --out data_new/property_occurrence_prop435k_split_dedup_sling/${fid} > log/sling_${fid} 2>&1 &
done
