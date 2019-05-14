#!/usr/bin/env python


from typing import Dict
from collections import defaultdict
import argparse, os, random
from itertools import combinations
from tqdm import tqdm
import numpy as np
from wikiutil.property import read_subprop_file, get_all_subtree, read_prop_occ_file, \
    get_is_sibling, print_subtree, read_subgraph_file, PropertySubtree
from wikiutil.data import read_emb_ids, filter_prop_occ_by_subgraph_and_emb, save_emb_ids


def traverse_subtree(subtree):
    yield subtree[0]
    for c in subtree[1]:
        yield from traverse_subtree(c)


def split_within_subtree(subtree, tr, dev, te):
    ''' split the subtree by spliting each tir into train/dev/test set '''
    siblings = [c[0] for c in subtree[1]]
    tr = int(len(siblings) * tr)
    dev = int(len(siblings) * dev)
    te = len(siblings) - tr - dev
    test = siblings[tr + dev:]
    dev = siblings[tr:tr + dev]
    train = siblings[:tr]
    if len(train) > 0 and len(dev) > 0 and len(test) > 0:
        yield train, dev, test
    for c in subtree[1]:
        split_within_subtree(c, tr, dev, te)


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
                        help='max subgraph sampled for each property')
    parser.add_argument('--method', type=str, default='by_tree', choices=['by_tree', 'within_tree'])
    args = parser.parse_args()

    '''
    python prep_data.py --prop_file data/subprops.txt --prop_dir data/property_occurrence_subtree/ \
        --subgraph_file data/property_occurrence_subtree.subgraph \
        --emb_file ~/tir1/data/wikidata/wikidata_translation_v1.tsv.id.QP \
        --out_dir data/analogy_dataset/test/ --method within_tree
    '''

    random.seed(2019)
    np.random.seed(2019)

    subprops = read_subprop_file(args.prop_file)
    pid2plabel = dict(p[0] for p in subprops)
    subtrees, isolate = get_all_subtree(subprops)

    prop2treeid = dict((p, i) for i, subtree in enumerate(subtrees) for p in traverse_subtree(subtree))

    # all the property ids that have been crawled
    all_propids = set()
    for root, dirs, files in os.walk(args.prop_dir):
        for file in files:
            if file.endswith('.txt'):  # do not use order file
                all_propids.add(file.rsplit('.', 1)[0])

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
        test_prop = [p for t in test_subtrees for p in traverse_subtree(t)
                     if p in all_propids]
        dev_prop = [p for t in dev_subtrees for p in traverse_subtree(t)
                    if p in all_propids and p not in test_prop]
        train_prop = [p for t in train_subtrees for p in traverse_subtree(t)
                      if p in all_propids and p not in dev_prop and p not in test_prop]

        print('totally {} subtrees, train {} /dev {} /test {}'.format(
            len(subtrees), len(train_subtrees), len(dev_subtrees), len(test_subtrees)))

        # save all subtrees
        for tree_split_name in ['train_subtrees', 'dev_subtrees', 'test_subtrees']:
            with open(os.path.join(args.out_dir, '.'.join(tree_split_name.split('_'))), 'w') as fout:
                tree_split = eval(tree_split_name)
                for st in tree_split:
                    fout.write(print_subtree(st, pid2plabel) + '\n\n')

    elif args.method == 'within_tree':
        # split each tir in a subtree into train/dev/test
        train_prop, dev_prop, test_prop = [], [], []
        for subtree in subtrees:
            for train_p, dev_p, test_p in split_within_subtree(subtree, tr, dev, te):
                train_prop.extend([p for p in train_p if p in all_propids])
                dev_prop.extend([p for p in dev_p if p in all_propids])
                test_prop.extend([p for p in test_p if p in all_propids])

    # remove duplicates
    train_prop = list(set(train_prop))
    dev_prop = list(set(dev_prop))
    test_prop = list(set(test_prop))
    print('totally {} properties, inititally train {} /dev {} /test {}'.format(
        len(all_propids), len(train_prop), len(dev_prop), len(test_prop)))

    # load subgraph and emb for existence check
    subgraph_dict = read_subgraph_file(args.subgraph_file)
    #save_emb_ids(args.emb_file, args.emb_file + '.id')
    emb_set = read_emb_ids(args.emb_file)

    # get multiple occurrence for each property
    is_sibling = get_is_sibling(subprops)
    final_prop_split: Dict[str, str] = defaultdict(lambda: '*')
    for prop_split_name in ['train_prop', 'dev_prop', 'test_prop']:
        prop_split = eval(prop_split_name)

        p2occs = {}
        for p in prop_split:
            occs = read_prop_occ_file(os.path.join(args.prop_dir, p + '.txt'), filter=True)
            occs = filter_prop_occ_by_subgraph_and_emb(p, occs, subgraph_dict, emb_set)  # check existence
            if len(occs) == 0:
                continue  # skip empty property
            occs = np.random.permutation(occs)[:args.max_occ_per_prop]
            p2occs[p] = occs

        print('{} out of {} property pass existence check'.format(len(p2occs), len(prop_split)))

        # save properties for this split
        with open(os.path.join(args.out_dir, '.'.join(prop_split_name.split('_'))), 'w') as fout:
            fout.write('\n'.join(map(lambda x: '\t'.join(x), [(p, pid2plabel[p]) for p in p2occs])) + '\n')

        # save to final split
        for p in p2occs:
            final_prop_split[p] = prop_split_name

        def get_all_pairs(p1, p2):
            for p1o in p2occs[p1]:
                for p2o in p2occs[p2]:
                    # yield 'p1 head tail', 'p2 head tail'
                    yield ' '.join([p1] + list(p1o)), ' '.join([p2] + list(p2o))

        data_filename = os.path.join(args.out_dir, prop_split_name.split('_')[0] + '.pointwise')
        with open(data_filename, 'w') as fout:
            print(data_filename)
            for p1, p2 in tqdm(combinations(p2occs.keys(), 2)):
                if (p1, p2) in is_sibling:  # positive pair
                    for p1o, p2o in get_all_pairs(p1, p2):
                        fout.write('{}\t{}\t{}\n'.format(1, p1o, p2o))
                else:  # negative pair
                    for p1o, p2o in get_all_pairs(p1, p2):
                        fout.write('{}\t{}\t{}\n'.format(0, p1o, p2o))

    if args.method == 'within_tree':
        subtrees_cls = [PropertySubtree(st) for st in subtrees]
        subtrees_cls = [st for st in subtrees_cls
                        if len(set(st.nodes) & set(final_prop_split.keys())) > 0]
        # concat label and split
        final_prop_split = dict((p, pid2plabel[p] + ' ' + final_prop_split[p].upper())
                                for p in pid2plabel)
        # save subtrees for this split
        with open(os.path.join(args.out_dir, 'within_tree_split.txt'), 'w') as fout:
            for st in subtrees_cls:
                fout.write(st.self_print_subtree(final_prop_split) + '\n\n')
