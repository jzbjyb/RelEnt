#!/usr/bin/env python


from typing import List, Tuple, Dict
import argparse, json, os, time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from random import shuffle
import re
from copy import deepcopy
import shutil
import pickle
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
from wikiutil.property import get_sub_properties, read_subprop_file, get_all_subtree, \
    hiro_subgraph_to_tree_dict, tree_dict_to_adj, read_prop_occ_file, PropertyOccurrence, read_subgraph_file, \
    read_prop_occ_file_from_dir, property_split
from wikiutil.util import read_emb_ids, load_tsv_as_dict, load_tsv_as_list, read_embeddings_from_text_file
from wikiutil.wikidata_query_service import get_property_occurrence
from wikiutil.textual_relation import WikipediaDataset

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
        with open(os.path.join(args.out, 'stat'), 'w') as fout:
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


def prop_occur_ana(args, check_existence=False, show_detail=False):
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
                                     #min_occ_per_prop=1000,
                                     #populate_method='combine_child',
                                     #subtrees=subtrees)
    print('property occurrence count:')
    p2count = [(x[0], len(x[1])) for x in poccs.pid2occs.items()]
    print(sorted(p2count, key=lambda x: -x[1]))
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
            if show_detail:
                print('{}\t{}'.format(child, len(child_occs)))
            for ancestor in ancestors:
                if ancestor not in subtree_pids:
                    continue
                anc_occs = pid2occs[ancestor]
                # we only care about the portion of the children occurrences that are included by the ancestor
                overlap_ratio = len(child_occs & anc_occs) / len(child_occs)
                overlaps.append((child, ancestor, len(child_occs), len(anc_occs), overlap_ratio))
                if show_detail:
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


def wikidata_populate(args):
    subprop_file, triple_file = args.inp.split(':')
    subprops = read_subprop_file(subprop_file)
    subtrees, isolate = get_all_subtree(subprops)
    child2ancestors = defaultdict(set)
    for subtree in subtrees:
        for c, ancs in subtree.traverse(return_ancestors=True):
            child2ancestors[c] |= set(ancs)
    print('average number of ancestors {}'.format(np.mean([len(child2ancestors[c]) for c in child2ancestors])))
    hash_set = set()
    with open(triple_file, 'r') as fin, open(args.out, 'w') as fout:
        for l in tqdm(fin, total=270306417):
            s, p, o = l.strip().split('\t')
            ps = set()
            if p in child2ancestors:
                ps = child2ancestors[p]
            ps.add(p)
            for p in ps:
                st = '\t'.join([s, p, o])
                if st not in hash_set:
                    fout.write(st + '\n')
                    hash_set.add(st)


def filter_ontology(args):
    count = 0
    with open(args.inp, 'r') as fin, open(args.out, 'w') as fout:
        for lc, line in tqdm(enumerate(fin), ncols=80, desc='Preparing triples', total=270306417):
            subj, rel, obj = line.strip().split('\t')
            if rel != 'P279' and rel != 'P31':  # only focus on "subclass of" and "instance of"
                continue
            count += 1
            fout.write(line)
    print('total number of triples involving subclass of: {}'.format(count))


def build_ontology(args, top_level=2):
    adjs = []
    id2ind = defaultdict(lambda: len(id2ind))
    parent2chids = defaultdict(lambda: [])
    child2parents = defaultdict(lambda: [])
    instof = defaultdict(lambda: [])
    all_inst = set()
    with open(args.inp, 'r') as fin:
        for i, l in tqdm(enumerate(fin)):
            subj, rel, obj = l.strip().split('\t')
            if rel == 'P279':
                adjs.append((id2ind[subj], id2ind[obj]))
                parent2chids[obj].append(subj)
                child2parents[subj].append(obj)
            elif rel == 'P31':
                instof[subj].append(obj)
                all_inst.add(obj)
    print('totally {} entities'.format(len(id2ind)))
    roots = set(id2ind.keys()) - set(child2parents.keys())
    leaves = set(id2ind.keys()) - set(parent2chids.keys())
    print('totally {} roots, {} leaves'.format(len(roots), len(leaves)))
    print('collect all top {} level nodes ...'.format(top_level))
    top_nodes = set()
    root2count = {}
    for root in roots:
        went = dfs_collect(root, parent2chids, depth=top_level)
        reduce_top_level = top_level
        # a heuristic to restrict the number of entities
        # (several biomed-related roots have large number of childrens)
        while len(went) >= 1000:
            reduce_top_level -= 1
            went = dfs_collect(root, parent2chids, depth=reduce_top_level)
        top_nodes.add(root)
        top_nodes.update(went)
        root2count[root] = len(went)
    print(sorted(root2count.items(), key=lambda x: -x[1])[:20])
    print('{} top-level nodes collected'.format(len(top_nodes)))
    print('find the nearest top node for each node in the ontology ...')
    node2dest: Dict[str, Tuple] = {}
    for leaf in tqdm(leaves):
        dfs_find(leaf, child2parents, destination=top_nodes, node2dest=node2dest, hist=set())
    for top_node in top_nodes:  # add top nodes
        node2dest[top_node] = (top_node, 0)
    all_subclass = set(node2dest.keys())
    not_attached = 0
    node2dest_inst: Dict[str, Tuple] = {}
    for inst in tqdm(all_inst):  # add instances
        # return first reached node in all_subclass or the top node
        node, find = dfs_find_quick(inst, instof, all_subclass, node2dest_inst, hist=set())
        if find:
            node2dest[inst] = node2dest[node]
        else:
            not_attached += 1
            node2dest[inst] = (node, -1)
    print('{} instance not attached on subclass'.format(not_attached))
    #assert len(node2dest) == len(id2ind), 'not all nodes are visited'
    with open(args.out, 'w') as fout:
        for node, (dest, depth) in node2dest.items():
            fout.write('{}\t{}\t{}\n'.format(node, dest, depth))


def dfs_find_quick(root, path: Dict[str, List], destination: set, node2dest: Dict[str, Tuple], hist=set()):
    if root in node2dest:
        return node2dest[root]
    if root in hist:  # cycle
        return root, False
    if root in destination:
        node2dest[root] = (root, True)
        return root, True
    if root not in path:
        node2dest[root] = (root, False)
        return root, False
    else:
        new_hist = deepcopy(hist)
        new_hist.add(root)
        for c in path[root]:
            node, find = dfs_find_quick(c, path, destination, node2dest, hist=new_hist)
            if find:
                node2dest[root] = (node, find)
                return node, find
        node2dest[root] = (node, False)
        return node, False


def dfs_find(root, path: Dict[str, List], destination: set, node2dest: Dict[str, Tuple], hist=set()):
    is_dest = False
    if root in hist:  # cycle
        return None, None
    if root in node2dest:
        return node2dest[root]
    if root in destination:
        is_dest = True  # still need go deeper, so don't return here
    if root not in path and not is_dest:
        raise Exception('cannot find destination for {}'.format(root))
    nearest_dep, nearest_dest = 10000, None
    next_hist = deepcopy(hist)
    next_hist.add(root)
    if root in path:
        for c in path[root]:
            if c in node2dest:
                dest, dep = node2dest[c]  # avoid multiple visiting
            else:
                dest, dep = dfs_find(c, path, destination, node2dest, hist=next_hist)
            if dest is not None and dep < nearest_dep:
                nearest_dep = dep
                nearest_dest = dest
    if is_dest:
        node2dest[root] = (root, 0)
        return root, 0
    if nearest_dest is None:
        return None, None
    node2dest[root] = (nearest_dest, nearest_dep + 1)
    return nearest_dest, nearest_dep + 1


def dfs_collect(root, path: Dict[str, List], depth=1) -> set:
    if depth <= 0 or root not in path:
        return {}
    went = set()
    for c in path[root]:
        went.add(c)
        went.update(dfs_collect(c, path, depth=depth-1))
    return went


def filter_triples(args):
    ''' note the all P31 "instance of" should also be kept '''
    triple_file, entity_file = args.inp.split(':')
    try:
        entities = set(load_tsv_as_dict(entity_file).keys())
    except:
        entities = set(load_tsv_as_list(entity_file, split=False))
    print('total number of frequent entities {}'.format(len(entities)))
    count = 0
    with open(triple_file, 'r') as tri_fin, \
            open(args.out, 'w') as fout:
        for lc, line in tqdm(enumerate(tri_fin), ncols=80, desc='Preparing triples', total=270306417):
            subj, rel, obj = line.strip().split('\t')
            # TODO: maybe we need to add "subclass of" as well because some entities don't have P31, e.g., Q204944
            if rel == 'P31' and subj in entities:
                count += 1
                fout.write(line)
            else:
                if subj not in entities or obj not in entities:
                    continue
                count += 1
                fout.write(line)
    print('total number of triples left {}'.format(count))


def downsample_by_property(args, ds_func=lambda x: int(min(x, np.sqrt(x) * 10))):
    ''' assume the property occurrence files are shuffled '''
    prop_occur_dir = args.inp
    inter_dir = args.out
    if os.path.exists(inter_dir):
        print('{} exists!'.format(inter_dir))
        exit(1)
    os.mkdir(inter_dir)
    def wc(fname):
        with open(fname, 'r') as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    ori_lines, ds_lines = 0, 0
    for root, dirs, files in os.walk(prop_occur_dir):
        for file in tqdm(files):
            if not file.endswith('.txt'):
                continue
            num_lines = wc(os.path.join(root, file))
            ori_lines += num_lines
            ds_num_lines = ds_func(num_lines)
            ds_lines += ds_num_lines
            os.system('head -n {} {} > {}'.format(
                ds_num_lines, os.path.join(root, file), os.path.join(inter_dir, file)))
    print('from #{} to #{}'.format(ori_lines, ds_lines))


def downsample_by_property_and_popularity(args, ds_func=lambda x: int(min(x, np.sqrt(x) * 10))):
    prop_occur_dir, entity2count, useful_props = args.inp.split(':')
    ds_dir = args.out
    if os.path.exists(ds_dir):
        print('{} exists!'.format(ds_dir))
        exit(1)

    # load entity counts
    entity2count = load_tsv_as_dict(entity2count, valuefunc=int)

    # load useful properties
    useful_props = set(load_tsv_as_dict(useful_props).keys())

    os.mkdir(ds_dir)
    ori_lines, ds_lines = 0, 0
    for root, dirs, files in os.walk(prop_occur_dir):
        for file in tqdm(files):
            if not file.endswith('.txt'):
                continue
            pid = file.rsplit('.', 1)[0]

            if pid not in useful_props:
                print('skip {} because not useful'.format(pid))
                continue

            ori_file = os.path.join(root, file)
            ds_file = os.path.join(ds_dir, file)

            # read all the occs
            occs: List[Tuple[str, str]] = read_prop_occ_file(
                ori_file, filter=True, contain_name=False, max_num=None)

            # rank by popularity
            occs = sorted(occs, key=lambda o: -entity2count[o[0]] - entity2count[o[1]])

            # compute
            ori_lines += len(occs)
            dl = ds_func(len(occs))
            ds_lines += dl

            # remove duplicates and shuffle
            ds_occs = set()
            for o in occs:
                ds_occs.add(o)
                if len(ds_occs) >= dl:
                    break
            ds_occs = list(ds_occs)
            shuffle(ds_occs)

            # output
            with open(ds_file, 'w') as fout:
                for o in ds_occs:
                    fout.write('{}\t{}\n'.format(*o))

    print('from #{} to #{} instances'.format(ori_lines, ds_lines))


def merge_poccs(args):
    poccs_dir = args.inp
    with open(args.out, 'w') as fout:
        for root, dirs, files in os.walk(poccs_dir):
            for file in tqdm(files):
                if not file.endswith('.txt'):
                    continue
                pid = file.rsplit('.', 1)[0]
                with open(os.path.join(root, file), 'r') as fin:
                    for l in fin:
                        s, o = l.strip().split('\t')
                        fout.write('{}\t{}\t{}\n'.format(s, pid, o))


def get_useless_props(args):
    subprops, poccs_dir = args.inp.split(':')
    subprops = read_subprop_file(subprops)

    miss, useful, useless = [], [], []
    for (pid, plabel), c in tqdm(subprops):
        label = -1
        for h, t in read_prop_occ_file_from_dir(
                pid, poccs_dir, filter=False, contain_name=True, max_num=None, use_order=True):
            if re.match('^Q[0-9]+$', h) and re.match('^Q[0-9]+$', t):
                label = 1
            else:
                label = 0
            break
        if label == -1:
            miss.append(pid)
        elif label == 1:
            useful.append(pid)
        elif label == 0:
            useless.append(pid)
    print('miss: {}, useful {}, useless {}'.format(len(miss), len(useful), len(useless)))
    return miss, useful, useless


def read_graph_link(filename):
    parent2chids = defaultdict(lambda: [])
    child2parents = defaultdict(lambda: [])
    with open(filename, 'r') as fin:
        for i, l in tqdm(enumerate(fin)):
            subj, rel, obj = l.strip().split('\t')
            if subj == 'Q35120':  # use "entity" as one of the root
                continue
            parent2chids[obj].append(subj)
            child2parents[subj].append(obj)
    return dict(parent2chids), dict(child2parents)


def get_partial_order(args):
    # load parent2child and child2parent
    p2c, c2p = read_graph_link(args.inp)
    ps = set(p2c.keys())
    cs = set(c2p.keys())
    all_nodes = ps | cs
    roots = all_nodes - cs
    went = set()
    last_layer = roots
    depth = 0
    node2dep = {}
    print('total {} nodes, with {} roots'.format(len(all_nodes), len(roots)))

    # bfs search
    while len(last_layer) > 0:
        print('depth: {}'.format(depth))
        next_layer = set()
        went.update(last_layer)
        for n in last_layer:
            node2dep[n] = depth
            if n in p2c:
                for c in p2c[n]:
                    can = True
                    for p in c2p[c]:
                        if p not in went:
                            can = False
                            break
                    if can and c not in went:
                        next_layer.add(c)
        last_layer = next_layer
        depth += 1

    # Use dfs to attach the remaining ones.
    # Because there are cycles, so the number in node2dep is not equal to the number in all_nodes.
    in_partial_order = set(node2dep.keys())
    other_node2dep = {}
    for other in tqdm(all_nodes - in_partial_order):
        dfs_find(other, c2p, destination=in_partial_order, node2dest=other_node2dep, hist=set())
    for k, (dest, dep) in other_node2dep.items():
        node2dep[k] = node2dep[dest] + dep

    with open(args.out, 'w') as fout:
        for k, v in sorted(node2dep.items(), key=lambda x: x[1]):
            fout.write('{}\t{}\n'.format(k, v))


def split_leaf_properties(args):
    pocc_dir, subgraph_dict, subprops, node2depth = args.inp.split(':')
    split_pocc_dir = args.out

    subprops = read_subprop_file(subprops)
    pid2plabel = dict(p[0] for p in subprops)
    pid2plabel_ = lambda x: pid2plabel[x.split('_', 1)[0]]
    subtrees, _ = get_all_subtree(subprops)
    subgraph_dict = read_subgraph_file(subgraph_dict)
    node2depth = load_tsv_as_dict(node2depth, valuefunc=int)

    # get all the leaves
    leaves = set()
    for subtree in subtrees:
        leaves.update(subtree.leaves)
    print('totally {} leaves'.format(len(leaves)))

    pid2splitcount = {}
    all_poccs = {}
    split_is_parent = set()
    ori_occ_count, new_occ_count = 0, 0

    # iterate over all the leaves to split
    for leaf in tqdm(leaves):
        this_pocccs = property_split(
            leaf, pocc_dir, subgraph_dict, node2depth, thres=0.01, debug=False, use_major_as_ori=False)
        if len(this_pocccs) <= 0:  # no occurrences for this property
            continue
        if leaf not in this_pocccs:  # no original property
            continue
        if len(this_pocccs) == 1:  # cannot be splitted
            continue

        # count and build parent-children relationship
        for k in this_pocccs:
            if k != leaf:
                split_is_parent.add((leaf, k))
                new_occ_count += len(this_pocccs[k])
        ori_occ_count += len(this_pocccs[leaf])
        pid2splitcount[leaf] = len(this_pocccs)

        # collect splits
        all_poccs.update(this_pocccs)

    print('totally {} parents'.format(len(pid2splitcount)))
    print('ori occ count: {}, new occ count: {}'.format(ori_occ_count, new_occ_count))
    print('pid -> split count:')
    print(sorted(pid2splitcount.items(), key=lambda x: -x[1]))

    pid2child: Dict[str, List[str]] = {}
    for (pid, plabel), cs in tqdm(subprops):
        from_file = os.path.join(pocc_dir, '{}.txt'.format(pid))
        to_file = os.path.join(split_pocc_dir, '{}.txt'.format(pid))
        if pid not in all_poccs:  # no split
            if os.path.exists(from_file):
                shutil.copy(from_file, to_file)
            pid2child[pid] = [c[0] for c in cs]
        else:
            parent = pid
            childs = [k for k in all_poccs if k.startswith(pid + '_')]
            for npid in [parent] + childs:
                with open(os.path.join(split_pocc_dir, '{}.txt'.format(npid)), 'w') as fout:
                    for h, t in all_poccs[npid]:
                        fout.write('{}\t{}\n'.format(h, t))
            pid2child[parent] = childs
            for c in childs:
                pid2child[c] = []
    with open(os.path.join(split_pocc_dir, 'subprops'), 'w') as fout:
        for k, v in pid2child.items():
            fout.write('{},{}\t{}\n'.format(
                k, pid2plabel_(k), '\t'.join(['{},{}'.format(c, pid2plabel_(c)) for c in v])))
    with open(os.path.join(split_pocc_dir, 'stat'), 'w') as fout:
        for pid, poccs in sorted(all_poccs.items(), key=lambda x: -len(x[1])):
            fout.write('{}\t{}\n'.format(pid, len(poccs)))


def make_hard_split(subprops_split: List,
                    emb: np.ndarray,
                    emb_id2ind: Dict[str, int],
                    method: str = 'middle'):
    # collect all the split childs
    p2c: Dict[str, set] = defaultdict(set)
    for (pid, plable), c in subprops_split:
        if pid.find('_') != -1:
            p2c[pid.split('_')[0]].add(pid)
    # find hard parent by similartiy
    long_tail_count, head_count = 0, 0
    p2hard: Dict[str, str] = {}
    for parent, childs in p2c.items():
        this_group = [parent] + list(childs)
        emb_for_this_group = np.array([emb[emb_id2ind[pid]] for pid in this_group], dtype=np.float32)
        sim = cosine_similarity(emb_for_this_group, emb_for_this_group)
        sim = np.mean(sim, axis=1)
        if method == 'min':
            outlier = this_group[np.argmin(sim)]
        elif method == 'middle':
            outlier = this_group[np.argsort(sim)[len(sim) // 2]]
        elif method.startswith('top_'):
            top_m = min(int(method.split('_', 1)[1]), len(sim))
            outlier = this_group[np.argsort(sim)[top_m]]
        else:
            raise NotImplemented
        p2hard[parent] = outlier
        if outlier == parent:
            long_tail_count += 1
        else:
            head_count += 1
    print('long tail outlier {}, head outlier {}'.format(long_tail_count, head_count))
    return p2hard


def replace_by_hard_split(args):
    subprops_split, emb = args.inp.split(':')

    emb_id2ind, emb = read_embeddings_from_text_file(emb, debug=False, emb_size=200, use_padding=True)
    subprops_split = read_subprop_file(subprops_split)
    pid2plabel = dict(p[0] for p in subprops_split)

    # get hard parent
    p2hard = make_hard_split(subprops_split, emb, emb_id2ind, method='middle')

    # save subprop
    hard2p = dict((v, k) for k, v in p2hard.items())
    for i, ((pid, plabel), c) in enumerate(subprops_split):
        if pid in p2hard and pid != p2hard[pid]:  # modify original parent
            for j, (cpid, cplabel) in enumerate(c):
                if cpid == p2hard[pid]:
                    c[j] = (pid, plabel)
                    break
            subprops_split[i] = ((p2hard[pid], pid2plabel[p2hard[pid]]), c)
        elif pid in hard2p and pid != hard2p[pid]:  # modify original child
            subprops_split[i] = ((hard2p[pid], pid2plabel[hard2p[pid]]), c)
        else:
            for j, (cpid, cplabel) in enumerate(c):  # modify original parent in other's leaves
                if cpid in p2hard:
                    c[j] = (p2hard[cpid], pid2plabel[p2hard[cpid]])

    with open(os.path.join(args.out, 'subprops_hard'), 'w') as fout:
        for (pid, plabel), c in subprops_split:
            fout.write('{},{}\t{}\n'.format(
                pid, plabel, '\t'.join(['{},{}'.format(cpid, cplabel) for cpid, cplabel in c])))


def link_entity_to_wikipedia(args, max_num_sent):
    triple_file, wikidata2wikipedia_file, wikipedia_data_dir = args.inp.split(':')
    # triple_file = 'data/split_merge_triples/property_occurrence_prop580k_split_shuf.tsv'
    # wikidata2wikipedia_file = 'data/wikipedia/wikidata_map.enwiki'
    # wikipedia_data_dir = 'data/wikipedia/en_extracted/'

    print('load wikidata ...')
    wikidata_ids = set()
    with open(triple_file, 'r') as fin:
        for l in fin:
            h, r, t = l.strip().split('\t')
            wikidata_ids.add(h)
            wikidata_ids.add(t)

    print('load wikipedia ...')
    wikidata2wikipedia_title = {}
    with open(wikidata2wikipedia_file, 'r') as fin:
        for l in fin:
            try:
                wd, wp = l.rstrip('\n').split(' ||| ')
                wp = wp.replace(' ', '_')
            except:
                print(l.rstrip('\n') + '$')
                continue
            wikidata2wikipedia_title[wd] = wp

    print('build ...')
    wp_dataset = WikipediaDataset(data_dir=wikipedia_data_dir)
    sid2sent, entity2sid = wp_dataset.build_entity2sent_for_wikidata(
        wikidata_ids=wikidata_ids, wikidata2wikipedia_title=wikidata2wikipedia_title, max_num_sent=max_num_sent)

    with open(os.path.join(args.out, 'sid2sent.pkl'), 'wb') as fout:
        pickle.dump(dict(sid2sent), fout)
    with open(os.path.join(args.out, 'entity2sid.pkl'), 'wb') as fout:
        pickle.dump(dict(entity2sid), fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('process wikidata property')
    parser.add_argument('--task', type=str,
                        choices=['subprop', 'build_tree', 'prop_occur_only',
                                 'prop_occur', 'prop_occur_all', 'prop_entities',
                                 'hiro_to_subgraph', 'prop_occur_ana',
                                 'wikidata_populate', 'filter_ontology', 'build_ontology',
                                 'filter_triples', 'downsample_by_property',
                                 'downsample_by_property_and_popularity', 'merge_poccs',
                                 'get_useless_props', 'get_partial_order',
                                 'split_leaf_properties', 'replace_by_hard_split',
                                 'link_entity_to_wikipedia'], required=True)
    parser.add_argument('--inp', type=str, required=None)
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
        only_subtree = False
        if only_subtree:
            # collect all the entities linked by the properties we are interested in
            subprops = read_subprop_file('data/subprops.txt')
            subtrees, isolate = get_all_subtree(subprops)
            pids = set()
            for subtree in subtrees:
                for pid in subtree.traverse():
                    pids.add(pid)
        else:
            subprops = read_subprop_file('data/subprops.txt')
            pids = set([p[0] for p, c in subprops])
        print('totally {} pids'.format(len(pids)))
        prop_entities(args, contain_name=False, pids=pids)
    elif args.task == 'hiro_to_subgraph':
        # convert the format hiro provides to list of tuples with a root node
        hiro_to_subgraph(args, max_hop=1)
    elif args.task == 'prop_occur_ana':
        # check entity-level overlap between parent and children properties
        prop_occur_ana(args, check_existence=False, show_detail=False)
    elif args.task == 'wikidata_populate':
        # populate wikidata by assuming all the parents properties holds for a triple
        wikidata_populate(args)
    elif args.task == 'filter_ontology':
        # filter wikidata triple file to keep only "subclass of" (P279) and "instance of" (P31)
        filter_ontology(args)
    elif args.task == 'build_ontology':
        # build ontology using the triples
        build_ontology(args, top_level=2)
    elif args.task == 'filter_triples':
        # filter triples to include only frequent entities
        filter_triples(args)
    elif args.task == 'downsample_by_property':
        # down-sample wikidata through properties
        downsample_by_property(args)
    elif args.task == 'downsample_by_property_and_popularity':
        # down-sample wikidata through properties
        ds_func = lambda x: int(min(x, np.sqrt(min(x, 1e5)) * 10))
        downsample_by_property_and_popularity(args, ds_func=ds_func)
    elif args.task == 'merge_poccs':
        # merge all the property occurrence files
        merge_poccs(args)
    elif args.task == 'get_useless_props':
        # how many properties are useful
        get_useless_props(args)
    elif args.task == 'get_partial_order':
        # get the partial order from ontology
        get_partial_order(args)
    elif args.task == 'split_leaf_properties':
        # split leaf properties by instance type of head and tail entity
        split_leaf_properties(args)
    elif args.task == 'replace_by_hard_split':
        # get hard parent for each split
        replace_by_hard_split(args)
    elif args.task == 'link_entity_to_wikipedia':
        link_entity_to_wikipedia(args, max_num_sent=1000)
