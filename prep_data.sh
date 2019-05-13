python prep_data.py --prop_file data/subprops.txt --prop_dir data/property_occurrence_subtree/ \
        --subgraph_file data/property_occurrence_subtree.subgraph \
        --emb_file ~/tir1/data/wikidata/wikidata_translation_v1.tsv.id.QP \
        --out_dir data/analogy_dataset/within_tree_532/ --method within_tree --train_dev_test 0.5:0.3:0.2
