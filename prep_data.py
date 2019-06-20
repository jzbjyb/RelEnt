#!/usr/bin/env python


from typing import Dict, Tuple
from collections import defaultdict
from random import shuffle
from operator import itemgetter
import argparse, os, random, time, pickle
from itertools import combinations, product
from tqdm import tqdm
import numpy as np
from wikiutil.property import read_subprop_file, get_all_subtree, \
    get_is_sibling, read_subgraph_file, get_is_parent, PropertyOccurrence, get_is_ancestor
from wikiutil.util import read_emb_ids, save_emb_ids, DefaultOrderedDict, load_tsv_as_list


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
                        choices=['by_tree', 'within_tree', 'by_entail', 'by_entail-n_way', 'by_entail-overlap'])
    parser.add_argument('--property_population', action='store_true',
                        help='parent properties are composed by children properties')
    parser.add_argument('--contain_train', action='store_true',
                        help='whether dev and test contain training properties')
    parser.add_argument('--num_sample', type=int, default=10000,
                        help='number of pairs to sample/sample for each property pair/property')
    parser.add_argument('--load_split', action='store_true',
                        help='directly load train/dev/test and parent (if any) properties from out_dir')
    parser.add_argument('--allow_empty_split', action='store_true',
                        help='whether empty split is allowed. used in within_tree and by_entail and by_entail-n_way')
    parser.add_argument('--filter_test', action='store_true', help='whether to remove test pid before population')
    args = parser.parse_args()

    random.seed(2019)
    np.random.seed(2019)

    subprops = read_subprop_file(args.prop_file)
    pid2plabel = dict(p[0] for p in subprops)
    subtrees, isolate = get_all_subtree(subprops)
    subtree_pids = set()  # only consider properties in subtrees
    [subtree_pids.add(pid) for subtree in subtrees for pid in subtree.traverse()]

    prop2treeid = dict((p, i) for i, subtree in enumerate(subtrees) for p in subtree.traverse())

    ## all the property ids that have occurrence
    all_propids = set()
    for root, dirs, files in os.walk(args.prop_dir):
        for file in files:
            if file.endswith('.txt') or file.endswith('.txt.order'):  # also use order file
                pid = file.split('.', 1)[0]
                if pid in subtree_pids:
                    all_propids.add(pid)

    ## load subgraph and emb for existence check
    subgraph_dict = read_subgraph_file(args.subgraph_file)
    #save_emb_ids(args.emb_file, args.emb_file + '.id')
    emb_set = read_emb_ids(args.emb_file)
    if args.property_population:
        if args.load_split and os.path.exists(os.path.join(args.out_dir, 'poccs.pickle')):
            print('load preprocessed property occs')
            with open(os.path.join(args.out_dir, 'poccs.pickle'), 'rb') as fin:
                poccs = PropertyOccurrence(pickle.load(fin),
                                           num_occ_per_subgraph=args.num_occ_per_subgraph)
        else:
            filter_pids = None
            if args.filter_test:
                filter_pids = set(map(itemgetter(0), 
                    load_tsv_as_list(os.path.join(args.out_dir, 'test.prop'))))
            poccs = PropertyOccurrence.build(sorted(all_propids), args.prop_dir,
                                             subgraph_dict=subgraph_dict,
                                             emb_set=emb_set,
                                             max_occ_per_prop=args.max_occ_per_prop,
                                             num_occ_per_subgraph=args.num_occ_per_subgraph,
                                             min_occ_per_prop=None,
                                             populate_method='top_down',
                                             subtrees=subtrees,
                                             filter_pids=filter_pids)
    else:
        # min_occ_per_prop is not used because some parent property (e.g., P3342: significant person)
        # is unexpectedly small, and obviously we don't want to loss any parent properties.
        poccs = PropertyOccurrence.build(sorted(all_propids), args.prop_dir,
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
            for train_p, dev_p, test_p in subtree.split_within(
                    tr, dev, te, filter_set=all_propids, allow_empty_split=args.allow_empty_split):
                train_prop.extend(train_p)
                dev_prop.extend(dev_p)
                test_prop.extend(test_p)

    elif args.method in {'by_entail', 'by_entail-n_way', 'by_entail-overlap'}:
        # split each tir in a subtree into train/dev/test by viewing parent property as label
        if args.load_split:
            print('load existing splits ...')
            parent_prop = list(map(itemgetter(0), load_tsv_as_list(os.path.join(args.out_dir, 'label2ind.txt'))))
            train_prop = list(map(itemgetter(0), load_tsv_as_list(os.path.join(args.out_dir, 'train.prop'))))
            dev_prop = list(map(itemgetter(0), load_tsv_as_list(os.path.join(args.out_dir, 'dev.prop'))))
            test_prop = list(map(itemgetter(0), load_tsv_as_list(os.path.join(args.out_dir, 'test.prop'))))
        else:
            parent_prop, train_prop, dev_prop, test_prop = [], [], [], []
            for subtree in subtrees:
                for parent, train_p, dev_p, test_p in subtree.split_within(
                        tr, dev, te, return_parent=True, filter_set=all_propids,
                        allow_empty_split=args.allow_empty_split):
                    if parent in all_propids:
                        parent_prop.append(parent)
                        train_prop.extend(train_p)
                        dev_prop.extend(dev_p)
                        test_prop.extend(test_p)
            # remove duplicates in parent properties, which will be used as labels
            parent_prop = list(set(parent_prop))
        print('totally {} parent properties (labels)'.format(len(parent_prop)))

    # remove duplicates and avoid overlap
    test_prop = list(set(test_prop))
    dev_prop = list(set(dev_prop) - set(test_prop))
    train_prop = list(set(train_prop) - set(dev_prop) - set(test_prop))
    print('totally {} properties, train {} /dev {} /test {}'.format(
        len(all_propids), len(train_prop), len(dev_prop), len(test_prop)))

    ## get multiple occurrence for each property
    is_sibling = get_is_sibling(subprops)
    is_parent = get_is_parent(subprops)
    is_ancestor = get_is_ancestor(subtrees)
    final_prop_split: Dict[str, str] = defaultdict(lambda: '*')
    label2id = DefaultOrderedDict(lambda: len(label2id))  # collect labels
    for prop_split_name in ['train_prop', 'dev_prop', 'test_prop']:
        prop_split = eval(prop_split_name)

        # save properties for this split
        with open(os.path.join(args.out_dir, '.'.join(prop_split_name.split('_'))), 'w') as fout:
            fout.write('\n'.join(map(lambda x: '\t'.join(x), [(p, pid2plabel[p]) for p in prop_split])) + '\n')

        # save to final split
        for p in prop_split:
            final_prop_split[p] = prop_split_name

        def format_occs(pid, poccs):
            # "pid _ occ1_head _ occ1_tail _ occ2_head _ occ2_tail ..." where _ is space
            return '{} {}'.format(pid, ' '.join(map(lambda occ: ' '.join(occ), poccs)))

        def get_all_pairs(pid1, pid2, pid1_num=1, pid2_num=1):
            for p1occs, p2occs in poccs.get_all_pairs(
                    pid1, pid2, args.num_sample, sam_for_pid1=pid1_num, sam_for_pid2=pid2_num):
                yield format_occs(pid1, p1occs), format_occs(pid2, p2occs)

        def get_all_occs(pid):
            for occs in poccs.get_all_occs(pid, args.num_sample):
                yield format_occs(pid, occs)

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
                _get_all_pairs = lambda p1, p2: get_all_pairs(p1, p2)
            elif args.method in {'by_entail'}:
                def pair_iter():
                    for parent, child in product(parent_prop, prop_split):
                        if parent == child:
                            continue
                        yield parent, child
                def pair_label_iter():
                    for p1, p2 in pair_iter():
                        yield p1, p2, int((p1, p2) in is_ancestor)  # TODO: use ancestor as positive examples?
                _get_all_pairs = lambda p1, p2: get_all_pairs(p1, p2, 5, 1)  # TODO: parent use 5 times occurrence
            with open(data_filename, 'w') as fout:
                for p1, p2, label in tqdm(pair_label_iter()):
                    for p1o, p2o in _get_all_pairs(p1, p2):
                        fout.write('{}\t{}\t{}\n'.format(label, p1o, p2o))

        # generate n way classification data
        elif args.method in {'by_entail-n_way'}:
            data_filename = os.path.join(args.out_dir, prop_split_name.split('_')[0] + '.nway')
            print(data_filename)
            with open(data_filename, 'w') as fout:
                for p in prop_split:
                    labels = [parent for parent in parent_prop if (parent, p) in is_parent]
                    if len(labels) != 1:
                        print('parents of {} are {}'.format(p, labels))
                    for label in labels:
                        for po in get_all_occs(p):
                            fout.write('{}\t{}\n'.format(label2id[label], po))
            print('{} labels up to now'.format(len(label2id)))

        elif args.method in {'by_entail-overlap'}:
            with open(os.path.join(args.out_dir, 'poccs.pickle'), 'wb') as fout:
                pickle.dump(poccs.pid2occs, fout)


    if args.method in {'within_tree', 'by_entail', 'by_entail-n_way', 'by_entail-overlap'}:
        subtrees_remains = [st for st in subtrees if len(set(st.nodes) & set(final_prop_split.keys())) > 0]
        # concat label and split
        final_prop_split = dict((p, pid2plabel[p] + ' ' + final_prop_split[p].upper()) for p in pid2plabel)
        # save subtrees for this split
        with open(os.path.join(args.out_dir, 'within_tree_split.txt'), 'w') as fout:
            for st in subtrees_remains:
                fout.write(st.print(final_prop_split) + '\n\n')

    if args.method in {'by_entail-n_way'}:
        # save parent properties (required because these are classification labels)
        with open(os.path.join(args.out_dir, 'label2ind.txt'), 'w') as fout:
            for label, ind in label2id.items():
                fout.write('{}\t{}\n'.format(label, ind))

    elif args.method in {'by_entail', 'by_entail-overlap'}:
        # save parent properties (optional)
        with open(os.path.join(args.out_dir, 'label2ind.txt'), 'w') as fout:
            for ind, prop in enumerate(parent_prop):
                fout.write('{}\t{}\n'.format(prop, ind))
