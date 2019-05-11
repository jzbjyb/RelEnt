#!/usr/bin/env python


import argparse, os, random
from itertools import combinations
from tqdm import tqdm
import numpy as np
from wikiutil.property import read_subprop_file, get_all_subtree, read_prop_occ_file, get_is_sibling


if __name__ == '__main__':
    parser = argparse.ArgumentParser('prepare data for analogy learning')
    parser.add_argument('--prop_file', type=str, required=True,
                        help='property file that specifies direct subproperties')
    parser.add_argument('--prop_dir', type=str, required=True,
                        help='directory of the property occurrence file')
    parser.add_argument('--subgraph_file', type=str, required=False,  # TODO: add existence check
                        help='entity subgraph file.')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--train_dev_test', type=str, default='0.8:0.1:0.1')
    parser.add_argument('--max_occ_per_prop', type=int, default=5,
                        help='max subgraph sampled for each property')
    args = parser.parse_args()

    random.seed(2019)
    np.random.seed(2019)

    subprops = read_subprop_file(args.prop_file)
    subtrees, isolate = get_all_subtree(subprops)

    def traverse_subtree(subtree):
        yield subtree[0]
        for c in subtree[1]:
            yield from traverse_subtree(c)

    prop2treeid = dict((p, i) for i, subtree in enumerate(subtrees) for p in traverse_subtree(subtree))

    # all the property ids that have been crawled
    all_propids = set()
    for root, dirs, files in os.walk(args.prop_dir):
        for file in files:
            if file.endswith('.txt'):  # do not use order file
                all_propids.add(file.rsplit('.', 1)[0])

    # split prop in train/dev/test (split subtree to avoid overlap)
    num_subtrees = len(subtrees)
    np.random.shuffle(subtrees)
    tr, dev, te = list(map(float, args.train_dev_test.split(':')))
    tr = int(num_subtrees * tr)
    dev = int(num_subtrees * dev)
    te = num_subtrees - tr - dev

    # avoid overlap between subtrees
    test_prop = [p for t in subtrees[tr + dev:] for p in traverse_subtree(t)
                 if p in all_propids]
    dev_prop = [p for t in subtrees[tr:tr + dev] for p in traverse_subtree(t)
                if p in all_propids and p not in test_prop]
    train_prop = [p for t in subtrees[:tr] for p in traverse_subtree(t)
                  if p in all_propids and p not in dev_prop and p not in test_prop]

    print('totally {} properties, train {} /dev {} /test {}'.format(
        len(all_propids), len(train_prop), len(dev_prop), len(test_prop)))

    # save all properties
    for prop_split_name in ['train_prop', 'dev_prop', 'test_prop']:
        with open(os.path.join(args.out_dir, '.'.join(prop_split_name.split('_'))), 'w') as fout:
            fout.write('\n'.join(eval(prop_split_name)) + '\n')

    # get multiple occurrence for each property
    is_sibling = get_is_sibling(subprops)
    for prop_split_name in ['train_prop', 'dev_prop', 'test_prop']:
        prop_split = eval(prop_split_name)

        p2occs = {}
        for p in prop_split:
            occs = read_prop_occ_file(os.path.join(args.prop_dir, p + '.txt'), filter=True)
            occs = np.random.permutation(occs)[:args.max_occ_per_prop]
            p2occs[p] = occs

        def get_all_pairs(p1, p2):
            for p1o in p2occs[p1]:
                for p2o in p2occs[p2]:
                    # yield 'p1 head tail', 'p2 head tail'
                    yield ' '.join([p1] + list(p1o)), ' '.join([p2] + list(p2o))

        with open(os.path.join(args.out_dir, prop_split_name.split('_')[0] + '.pointwise'), 'w') as fout:
            for p1, p2 in tqdm(combinations(prop_split, 2)):
                if (p1, p2) in is_sibling:  # positive pair
                    for p1o, p2o in get_all_pairs(p1, p2):
                        fout.write('{}\t{}\t{}\n'.format(1, p1o, p2o))
                else:  # negative pair
                    for p1o, p2o in get_all_pairs(p1, p2):
                        fout.write('{}\t{}\t{}\n'.format(0, p1o, p2o))
