#!/usr/bin/env python


import argparse, json
from collections import defaultdict
import numpy as np
from wikiutil.property import get_sub_properties, read_subprop_file, get_subtree, print_subtree, get_depth


def subprop(args):
    all_props = []
    with open(args.inp, 'r') as fin:
        props = json.load(fin)
        for prop in props:
            pid, plabel = prop['id'], prop['label']
            all_props.append((pid, plabel))
    all_props = sorted(all_props, key=lambda x: int(x[0][1:]))
    num_parent = 0
    num_pairs = 0
    with open(args.out, 'w') as fout:
        for pid, plabel in all_props:
            subs = get_sub_properties(pid)
            if len(subs) > 0:
                num_parent += 1
                num_pairs += len(subs) * (len(subs) - 1)
            ln = '{},{}\t{}\n'.format(pid, plabel, 
                '\t'.join(map(lambda x: ','.join(x), subs)))
            fout.write(ln)
    print('{} props'.format(len(all_props)))
    print('{} parents'.format(num_parent))
    print('{} pairs'.format(num_pairs))


def build_tree(args):
    subprops = read_subprop_file(args.inp)
    num_prop = len(subprops)
    print('{} props'.format(num_prop))

    # get pid to plabel dict
    pid2plabel = dict(p[0] for p in subprops)

    # get parent link and children link
    parent_dict = defaultdict(lambda: [])
    child_dict = defaultdict(lambda: [])
    for p in subprops:
        parent_id = p[0][0]
        child_dict[parent_id] = [c[0] for c in p[1]]
        for c in p[1]:
            parent_dict[c[0]].append(parent_id)

    # construct tree for properties without parent
    subtrees, isolate = [], []
    for p in subprops:
        pid = p[0][0]
        if len(parent_dict[pid]) == 0:
            subtree = get_subtree(pid, child_dict)
            if len(subtree[1]) > 0:
                subtrees.append(subtree)
            else:
                isolate.append(subtree)

    print('{} subtree'.format(len(subtrees)))
    print('avg depth: {}'.format(np.mean([get_depth(s) for s in subtrees])))
    print('{} isolated prop'.format(len(isolate)))

    with open(args.out, 'w') as fout:
        fout.write('\n--- subtrees ---\n\n')
        for st in subtrees:
            fout.write(print_subtree(st, pid2plabel) + '\n\n')
        fout.write('\n--- isolated properties ---\n\n')
        for ip in isolate:
            fout.write('{}: {}\n'.format(ip[0], pid2plabel[ip[0]]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('process wikidata property')
    parser.add_argument('--task', type=str, choices=['subprop', 'build_tree'], required=True)
    parser.add_argument('--inp', type=str, required=True)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    if args.task == 'subprop':
        # extract sub-properties for all the properties in Wikidata
        subprop(args)
    elif args.task == 'build_tree':
        # build a tree-like structure using the sub-properties extracted
        build_tree(args)
