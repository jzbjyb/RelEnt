#!/usr/bin/env python


import argparse, json, os, time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from wikiutil.property import get_sub_properties, read_subprop_file, get_subtree, print_subtree, get_depth, \
    hiro_subgraph_to_tree_dict, tree_dict_to_adj
from wikiutil.wikidata_query_service import get_property_occurrence

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


def build_tree(args, use_return=False):
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

    if use_return:
        return subtrees, isolate

    with open(args.out, 'w') as fout:
        fout.write('\n--- subtrees ---\n\n')
        for st in subtrees:
            fout.write(print_subtree(st, pid2plabel) + '\n\n')
        fout.write('\n--- isolated properties ---\n\n')
        for ip in isolate:
            fout.write('{}: {}\n'.format(ip[0], pid2plabel[ip[0]]))


def prop_occur(args, only_isolate=True):
    subprops = read_subprop_file(args.inp)
    num_prop = len(subprops)
    print('{} props'.format(num_prop))

    _, isolate = build_tree(args, use_return=True)
    isolate = set(iso[0] for iso in isolate)

    for prop in subprops:
        pid = prop[0][0]
        if only_isolate and pid in isolate:
            continue
        try:
            try:
                rand = True
                results = get_property_occurrence(pid, limit=1000, timeout=30)
            except:
                time.sleep(60)
                rand = False
                results = get_property_occurrence(pid, limit=1000, timeout=30, fast=True)
            print('fetch {}, get {} occurrence'.format(pid, len(results['results']['bindings'])))
            suffix = '' if rand else '.order'
            with open(os.path.join(args.out, pid + '.txt' + suffix), 'w') as fout, \
                    open(os.path.join(args.out, pid + '.raw' + suffix), 'w') as raw_fout:
                for result in results['results']['bindings']:
                    raw_fout.write('{}\n'.format(result))
                    if result['item']['type'] != 'uri' or result['value']['type'] != 'uri':
                        continue
                    item = result['item']['value'].rsplit('/')[-1]
                    value = result['value']['value'].rsplit('/')[-1]
                    item_label = result['itemLabel']['value']
                    value_label = result['valueLabel']['value']
                    fout.write('{}\t{}\t{}\t{}\n'.format(item, item_label, value, value_label))
            time.sleep(5)
        except Exception as e:
            print('exception at {}'.format(pid))
            time.sleep(60)


def prop_entities(args):
    cache = set()
    rand_num, non_rand_num = 0, 0
    with open(args.out, 'w') as fout:
        for root, dirs, files in os.walk(args.inp):
            for file in files:
                if file.endswith('.txt'):
                    rand_num += 1
                elif file.endswith('.txt.order'):
                    non_rand_num += 1
                else:
                    continue
                with open(os.path.join(root, file), 'r') as fin:
                    for l in fin:
                        hid, _, tid, _ = l.strip().split('\t')
                        if hid not in cache:
                            fout.write('{}\n'.format(hid))
                            cache.add(hid)
                        if tid not in cache:
                            fout.write('{}\n'.format(tid))
                            cache.add(tid)
    print('#random file {}, #non-random file {}'.format(rand_num, non_rand_num))


def hiro_to_subgraph(args, max_hop=1):
    eid_file, hiro_file = args.inp.split(':')
    with open(eid_file, 'r') as eid_fin, \
            open(hiro_file, 'r') as hiro_fin, \
            open(args.out, 'w') as fout:
        for i, l in tqdm(enumerate(eid_fin)):
            eid = l.strip()
            hiro_subg = json.loads(hiro_fin.readline().strip())
            tree_dict = hiro_subgraph_to_tree_dict(eid, hiro_subg, max_hop=max_hop)
            adj_list = tree_dict_to_adj(tree_dict)
            fout.write(eid + '\t')  # write root entity
            fout.write('\t'.join(map(lambda x: ' '.join(x), adj_list)))
            fout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('process wikidata property')
    parser.add_argument('--task', type=str,
                        choices=['subprop', 'build_tree', 'prop_occur_only',
                                 'prop_occur', 'prop_entities', 'hiro_to_subgraph'], required=True)
    parser.add_argument('--inp', type=str, required=True)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    if args.task == 'subprop':
        # extract sub-properties for all the properties in Wikidata
        subprop(args)
    elif args.task == 'build_tree':
        # build a tree-like structure using the sub-properties extracted
        build_tree(args)
    elif args.task == 'prop_occur_only':
        # get entities linked by properties in a subtree
        prop_occur(args, only_isolate=True)
    elif args.task == 'prop_occur':
        # get entities linked by all properties
        prop_occur(args, only_isolate=False)
    elif args.task == 'prop_entities':
        # collect all the entities linked by the properties we are interested in
        prop_entities(args)
    elif args.task == 'hiro_to_subgraph':
        # convert the format hiro provides to list of tuples with a root node
        hiro_to_subgraph(args, max_hop=1)
