#!/usr/bin/env python


from typing import Dict, Tuple
from collections import defaultdict
from random import shuffle
import argparse, os, random, time
from itertools import combinations, product
from tqdm import tqdm
import numpy as np
from wikiutil.property import read_subprop_file, get_all_subtree, read_prop_occ_file_from_dir, \
    get_is_sibling, read_subgraph_file, get_is_parent, PropertyOccurrence
from wikiutil.util import read_emb_ids, save_emb_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser('prepare data for analogy learning')
    parser.add_argument('--prop_file', type=str, required=True,
                        help='property file that specifies direct subproperties')
    parser.add_argument('--prop_dir', type=str, required=True,
                        help='directory of the property occurrence file')
    parser.add_argument('--subgraph_file', type=str, required=True,
                        help='entity subgraph file.')
    parser.add_argument('--emb_file', type=str, required=True,
                        help='embedding file.')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--train_dev_test', type=str, default='0.8:0.1:0.1')
    parser.add_argument('--max_occ_per_prop', type=int, default=5,
                        help='max occurrence kept for each property')
    parser.add_argument('--num_occ_per_subgraph', type=int, default=1,
                        help='number of occurrences used for each property subgraph')
    parser.add_argument('--method', type=str, default='by_tree',
                        choices=['by_tree', 'within_tree', 'by_entail'])
    parser.add_argument('--contain_train', action='store_true',
                        help='whether dev and test contain training properties')
    parser.add_argument('--num_per_prop_pair', type=int, default=10000,
                        help='number of pairs to sample for each property pair')
    args = parser.parse_args()

    random.seed(2019)
    np.random.seed(2019)

    subprops = read_subprop_file(args.prop_file)
    pid2plabel = dict(p[0] for p in subprops)
    subtrees, isolate = get_all_subtree(subprops)

    prop2treeid = dict((p, i) for i, subtree in enumerate(subtrees) for p in subtree.traverse())

    ## all the property ids that have been crawled
    all_propids = set()
    for root, dirs, files in os.walk(args.prop_dir):
        for file in files:
            if file.endswith('.txt') or file.endswith('.txt.order'):  # also use order file
                all_propids.add(file.split('.', 1)[0])

    ## load subgraph and emb for existence check
    subgraph_dict = read_subgraph_file(args.subgraph_file)
    #save_emb_ids(args.emb_file, args.emb_file + '.id')
    emb_set = read_emb_ids(args.emb_file)
    poccs = PropertyOccurrence.build(all_propids, args.prop_dir,
                                     subgraph_dict=subgraph_dict,
                                     emb_set=emb_set,
                                     max_occ_per_prop=args.max_occ_per_prop,
                                     num_occ_per_subgraph=args.num_occ_per_subgraph)

    print('{} out of {} property pass existence check'.format(len(poccs.pids), len(all_propids)))
    all_propids &= set(poccs.pids)

    ## split train/dev/test
    tr, dev, te = list(map(float, args.train_dev_test.split(':')))

    if args.method == 'by_tree':
        # split subtrees into train/dev/test
        num_subtrees = len(subtrees)
        np.random.shuffle(subtrees)
        tr = int(num_subtrees * tr)
        dev = int(num_subtrees * dev)
        te = num_subtrees - tr - dev
        test_subtrees = subtrees[tr + dev:]
        dev_subtrees = subtrees[tr:tr + dev]
        train_subtrees = subtrees[:tr]

        # avoid overlap between subtrees
        test_prop = [p for t in test_subtrees for p in t.traverse() if p in all_propids]
        dev_prop = [p for t in dev_subtrees for p in t.traverse() if p in all_propids]
        train_prop = [p for t in train_subtrees for p in t.traverse() if p in all_propids]

        print('totally {} subtrees, train {} /dev {} /test {}'.format(
            len(subtrees), len(train_subtrees), len(dev_subtrees), len(test_subtrees)))

        # save all subtrees
        for tree_split_name in ['train_subtrees', 'dev_subtrees', 'test_subtrees']:
            with open(os.path.join(args.out_dir, '.'.join(tree_split_name.split('_'))), 'w') as fout:
                tree_split = eval(tree_split_name)
                for st in tree_split:
                    fout.write(st.print(pid2plabel) + '\n\n')

    elif args.method == 'within_tree':
        # split each tir in a subtree into train/dev/test
        train_prop, dev_prop, test_prop = [], [], []
        for subtree in subtrees:
            for train_p, dev_p, test_p in subtree.split_within(tr, dev, te, filter_set=all_propids):
                train_prop.extend(train_p)
                dev_prop.extend(dev_p)
                test_prop.extend(test_p)

    elif args.method == 'by_entail':
        # split each tir in a subtree into train/dev/test by viewing parent property as label
        parent_prop, train_prop, dev_prop, test_prop = [], [], [], []
        for subtree in subtrees:
            for parent, train_p, dev_p, test_p in subtree.split_within(
                    tr, dev, te, return_parent=True, filter_set=all_propids):
                if parent in all_propids:
                    parent_prop.append(parent)
                    train_prop.extend(train_p)
                    dev_prop.extend(dev_p)
                    test_prop.extend(test_p)
        # remove duplicates in parent properties, which will be used as labels
        parent_prop = list(set(parent_prop))
        print('totoall {} parent properties (labels)'.format(len(parent_prop)))

    # remove duplicates and avoid overlap
    test_prop = list(set(test_prop))
    dev_prop = list(set(dev_prop) - set(test_prop))
    train_prop = list(set(train_prop) - set(dev_prop) - set(test_prop))
    print('totally {} properties, train {} /dev {} /test {}'.format(
        len(all_propids), len(train_prop), len(dev_prop), len(test_prop)))

    ## get multiple occurrence for each property
    is_sibling = get_is_sibling(subprops)
    is_parent = get_is_parent(subprops)
    final_prop_split: Dict[str, str] = defaultdict(lambda: '*')
    for prop_split_name in ['train_prop', 'dev_prop', 'test_prop']:
        prop_split = eval(prop_split_name)

        # save properties for this split
        with open(os.path.join(args.out_dir, '.'.join(prop_split_name.split('_'))), 'w') as fout:
            fout.write('\n'.join(map(lambda x: '\t'.join(x), [(p, pid2plabel[p]) for p in prop_split])) + '\n')

        # save to final split
        for p in prop_split:
            final_prop_split[p] = prop_split_name

        def get_all_pairs(pid1, pid2):
            def format(pid, poccs):
                # "pid _ occ1 _ occ1 _ occ2 _ occ2 ..." where _ is space
                return '{} {}'.format(pid, ' '.join(map(lambda occ: ' '.join(occ), poccs)))
            for p1occs, p2occs in poccs.get_all_pairs(pid1, pid2, args.num_per_prop_pair):
                yield format(pid1, p1occs), format(pid2, p2occs)

        # generate pair of property occurrences for binary classification
        if args.method in {'by_tree', 'within_tree', 'by_entail'}:
            data_filename = os.path.join(args.out_dir, prop_split_name.split('_')[0] + '.pointwise')
            print(data_filename)
            if args.method in {'by_tree', 'within_tree'}:
                def pair_iter():
                    yield from combinations(prop_split, 2)
                    if args.contain_train and (prop_split_name.startswith('dev_') or
                                               prop_split_name.startswith('test_')):
                        yield from product(prop_split, train_prop)
                def pair_label_iter():
                    for p1, p2 in pair_iter():
                        yield p1, p2, int((p1, p2) in is_sibling)
            elif args.method in {'by_entail'}:
                def pair_iter():
                    yield from product(parent_prop, prop_split)
                def pair_label_iter():
                    for p1, p2 in pair_iter():
                        yield p1, p2, int((p1, p2) in is_parent)
            with open(data_filename, 'w') as fout:
                for p1, p2, label in tqdm(pair_label_iter()):
                    for p1o, p2o in get_all_pairs(p1, p2):
                        fout.write('{}\t{}\t{}\n'.format(label, p1o, p2o))

    if args.method in {'within_tree', 'by_entail'}:
        subtrees_remains = [st for st in subtrees if len(set(st.nodes) & set(final_prop_split.keys())) > 0]
        # concat label and split
        final_prop_split = dict((p, pid2plabel[p] + ' ' + final_prop_split[p].upper()) for p in pid2plabel)
        # save subtrees for this split
        with open(os.path.join(args.out_dir, 'within_tree_split.txt'), 'w') as fout:
            for st in subtrees_remains:
                fout.write(st.print(final_prop_split) + '\n\n')
