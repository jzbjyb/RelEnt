#!/usr/bin/env python


from typing import List, Tuple, Dict
import argparse, json, os, time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from wikiutil.property import get_sub_properties, read_subprop_file, get_all_subtree, \
    hiro_subgraph_to_tree_dict, tree_dict_to_adj, read_prop_occ_file, PropertyOccurrence, read_subgraph_file
from wikiutil.util import read_emb_ids
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


def build_tree(args):
    subprops = read_subprop_file(args.inp)
    # get pid to plabel dict
    pid2plabel = dict(p[0] for p in subprops)
    # get all subtrees
    subtrees, isolate = get_all_subtree(subprops)

    with open(args.out, 'w') as fout:
        fout.write('\n--- subtrees ---\n\n')
        for st in subtrees:
            fout.write(st.print(pid2plabel) + '\n\n')
        fout.write('\n--- isolated properties ---\n\n')
        for ip in isolate:
            fout.write(ip.print(pid2plabel) + '\n\n')


def prop_occur(args, only_isolate=True):
    subprops = read_subprop_file(args.inp)
    num_prop = len(subprops)
    print('{} props'.format(num_prop))

    _, isolate = get_all_subtree(subprops)
    isolate = set(iso.nodes[0] for iso in isolate)

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


def prop_occur_all(args):
    rel_count_dict = defaultdict(lambda: 0)
    rel_file_dict = {}
    try:
        with open(args.inp, 'r') as fin:
            for i, l in tqdm(enumerate(fin)):
                subj, rel, obj = l.strip().split('\t')
                if not (subj.startswith('Q') and obj.startswith('Q') and rel.startswith('P')):
                    # only consider real entities and properties
                    continue
                if rel_count_dict[rel] == 0:
                    rel_file_dict[rel] = open(os.path.join(args.out, rel + '.txt'), 'w')
                rel_file_dict[rel].write('{}\t{}\n'.format(subj, obj))
                rel_count_dict[rel] += 1
        with open(os.path.join(args.out, 'stat.txt'), 'w') as fout:
            for pid, count in sorted(rel_count_dict.items(), key=lambda x: -x[1]):
                fout.write('{}\t{}\n'.format(pid, count))
    finally:
        print('closing files ...')
        for k, f in rel_file_dict.items():
            f.close()


def prop_entities(args, contain_name=True, pids=None):
    cache = set()
    rand_num, non_rand_num = 0, 0
    if pids:
        def file_iter():
            for pid in pids:
                file = os.path.join(args.inp, pid + '.txt')
                if os.path.exists(file):
                    yield file
    else:
        def file_iter():
            nonlocal rand_num, non_rand_num
            for root, dirs, files in os.walk(args.inp):
                for file in files:
                    if file.endswith('.txt'):
                        rand_num += 1
                    elif file.endswith('.txt.order'):
                        non_rand_num += 1
                    else:
                        continue
                    yield os.path.join(root, file)
    with open(args.out, 'w') as fout:
        for filepath in tqdm(file_iter()):
            occs = read_prop_occ_file(filepath, filter=True, contain_name=contain_name)
            for hid, tid in occs:
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
            if max_hop == 1:
                adj_list = []
                for node in hiro_subg:
                    plist, tid, depth, parent_eid = node
                    if depth > max_hop:
                        break
                    adj_list.append((eid, plist[-1], tid))
            else:
                tree_dict = hiro_subgraph_to_tree_dict(eid, hiro_subg, max_hop=max_hop)
                adj_list = tree_dict_to_adj(tree_dict)
            fout.write(eid + '\t')  # write root entity
            fout.write('\t'.join(map(lambda x: ' '.join(x), adj_list)))
            fout.write('\n')


def prop_occur_ana(args, check_existence=False):
    subprop_file, subgraph_file, emb_file, prop_occur_dir = args.inp.split(':')
    subprops = read_subprop_file(subprop_file)
    # get pid to plabel dict
    pid2plabel = dict(p[0] for p in subprops)
    # get all subtrees
    subtrees, isolate = get_all_subtree(subprops)
    subtree_pids = set()  # only consider properties in subtrees
    [subtree_pids.add(pid) for subtree in subtrees for pid in subtree.traverse()]
    if check_existence:
        # load subgraph and emb
        subgraph_dict = read_subgraph_file(subgraph_file)
        emb_set = read_emb_ids(emb_file)
    else:
        subgraph_dict = emb_set = None
    # get all occurrences
    poccs = PropertyOccurrence.build(sorted(subtree_pids), prop_occur_dir,
                                     subgraph_dict=subgraph_dict,
                                     emb_set=emb_set,
                                     max_occ_per_prop=1000000,
                                     num_occ_per_subgraph=1)
    subtree_pids &= set(poccs.pids)  # all the properties considered
    # (child, ancestor, child count, ancestor count, overlap ratio)
    overlaps: List[Tuple[str, str, int, int, float]] = []
    # turn property occurrences into set
    pid2occs = dict((k, set(v)) for k, v in poccs.pid2occs.items())
    for subtree in tqdm(subtrees):
        for child, ancestors in subtree.traverse(return_ancestors=True):
            if child not in subtree_pids:
                continue
            child_occs = pid2occs[child]
            print('{}\t{}'.format(child, len(child_occs)))
            for ancestor in ancestors:
                if ancestor not in subtree_pids:
                    continue
                anc_occs = pid2occs[ancestor]
                # we only care about the portion of the children occurrences that are included by the ancestor
                overlap_ratio = len(child_occs & anc_occs) / len(child_occs)
                overlaps.append((child, ancestor, len(child_occs), len(anc_occs), overlap_ratio))
                print('\t{}\t{}\t{}'.format(ancestor, len(anc_occs), overlap_ratio))
    overlaps = sorted(overlaps, key=lambda x: -x[-1])
    ol_by_ancestor: Dict[str, float] = defaultdict(list)
    for child, ancestor, _, _, ol in overlaps:
        ol_by_ancestor[ancestor].append(ol)
    ol_by_ancestor = dict((anc, np.mean(ol)) for anc, ol in ol_by_ancestor.items())
    with open(args.out, 'w') as fout:
        fout.write('--- children and ancestor property occurrences overlap ---\n\n')
        for overlap in overlaps:
            fout.write(' '.join(map(str, overlap)) + '\n')
        fout.write('\n\n--- ancestor property occurrences overlap ---\n\n')
        for anc, overlap in sorted(ol_by_ancestor.items(), key=lambda x: -x[1]):
            fout.write('{}\t{}\n'.format(anc, overlap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('process wikidata property')
    parser.add_argument('--task', type=str,
                        choices=['subprop', 'build_tree', 'prop_occur_only',
                                 'prop_occur', 'prop_occur_all', 'prop_entities',
                                 'hiro_to_subgraph', 'prop_occur_ana'], required=True)
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
    elif args.task == 'prop_occur_all':
        # get all entities linked by all properties using hiro's tuple file
        prop_occur_all(args)
    elif args.task == 'prop_entities':
        # collect all the entities linked by the properties we are interested in
        subprops = read_subprop_file('data/subprops.txt')
        subtrees, isolate = get_all_subtree(subprops)
        pids = set()
        for subtree in subtrees:
            for pid in subtree.traverse():
                pids.add(pid)
        print('totally {} pids'.format(len(pids)))
        prop_entities(args, contain_name=False, pids=pids)
    elif args.task == 'hiro_to_subgraph':
        # convert the format hiro provides to list of tuples with a root node
        hiro_to_subgraph(args, max_hop=1)
    elif args.task == 'prop_occur_ana':
        # check entity-level overlap between parent and children properties
        prop_occur_ana(args, check_existence=False)
