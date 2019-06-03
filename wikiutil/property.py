from typing import List, Tuple, Dict, Iterable, Union
from collections import defaultdict
from itertools import combinations
from random import shuffle
from copy import deepcopy
from tqdm import tqdm
import subprocess, re, os, random, functools, pickle
import numpy as np


def hiro_subgraph_to_tree_dict(root: str,
                               hiro_subg: List[Tuple[List[str], str, int]],
                               max_hop=1) -> Tuple[str, Dict[str, List[Tuple[str, Dict]]]]:
    # group all the entities by their depth
    # the items in the tuple are: relations, current entity id, depth, parent entity id
    hop_dict: Dict[int, List[Tuple[List[str], str, int, str]]] = defaultdict(lambda: [])
    for e in hiro_subg:
        depth = e[2]
        hop_dict[depth].append(e)
    # generate subgraph dict from small hop to large hop
    # recursive tree structure that cannot handle cycles
    # note that a property might point to multiple entities
    # TODO: hiro's graph is acyclic by its nature and might not be complete
    # TODO: hiro's graph is not complete comparing to the wikidata page and some properties are duplicate
    tree_dict: Tuple[str, Dict[str, List[Tuple[str, Dict]]]] = (root, {})
    for hop in range(max_hop):
        hop += 1
        if hop > 1:
            raise NotImplementedError  # TODO: cannot construct the subgraph from hiro's data structure
        for e in hop_dict[hop]:
            plist, eid, _, parent_eid = e
            trace = tree_dict[1]
            parent = None
            for p in plist[:-1]:
                parent = trace[p][0]
                trace = trace[p][1]
            if plist[-1] not in trace:
                trace[plist[-1]] = []
            trace[plist[-1]].append((eid, {}))
    return tree_dict


def tree_dict_to_adj(tree_dict: Tuple[str, Dict[str, List[Tuple[str, Dict]]]]) -> List[Tuple[str, str, str]]:
    # DFS to get all the connections used to construct adjacency matrix
    adjs: List[Tuple[str, str, str]] = []
    root = tree_dict[0]
    for p in tree_dict[1]:  # iterative through properites
        for c in tree_dict[1][p]:  # iterate through entities this property is pointing to
            adjs.append((root, p, c[0]))
            adjs.extend(tree_dict_to_adj(c))
    return adjs


def get_sub_properties(pid):
    output = subprocess.check_output(['wdtaxonomy', pid, '-f', 'csv'])
    output = output.decode('utf-8')
    subs = []
    for l in output.split('\n')[1:]:
        if l.startswith('-,'):  # first-level sub props
            l = l.split(',')
            subs.append((l[1], l[2].strip('"')))
    return subs


def read_prop_occ_file_from_dir(prop: str, dir: str, filter=False, contain_name=True, max_num=None, use_order=False):
    filepath = os.path.join(dir, prop + '.txt')
    filepath_order = os.path.join(dir, prop + '.txt.order')
    if os.path.exists(filepath):
        yield from read_prop_occ_file(filepath, filter=filter, contain_name=contain_name, max_num=max_num)
    elif use_order and os.path.exists(filepath_order):
        yield from read_prop_occ_file(filepath_order, filter=filter, contain_name=contain_name, max_num=max_num)
    else:
        return  # empty iterator


def read_prop_occ_file(filepath, filter=False, contain_name=True, max_num=None) -> Iterable[Tuple[str, str]]:
    count = 0
    # TODO: there might be duplicates in the file
    with open(filepath, 'r') as fin:
        for l in fin:
            if contain_name:
                hid, _, tid, _ = l.strip().split('\t')
            else:
                hid, tid = l.strip().split('\t')
            if filter and not re.match('^Q[0-9]+$', hid) or not re.match('^Q[0-9]+$', tid):
                # only keep entities
                continue
            yield (hid, tid)
            count += 1
            if max_num and count >= max_num:
                break


def get_prop_occ_by_minus(dir: str, main_prop: str, other_props: List[str], max_num: int = None):
    main_occs = read_prop_occ_file_from_dir(
        main_prop, dir, filter=False, contain_name=False, max_num=max_num, use_order=False)
    main_occs = set(main_occs)
    other_occs = set()
    for prop in other_props:
        occs = read_prop_occ_file_from_dir(
            prop, dir, filter=False, contain_name=False, max_num=max_num, use_order=False)
        other_occs.update(occs)
    not_contained = list(main_occs - other_occs)
    print('{} occs out of {} not contained in {}'.format(len(not_contained), len(main_occs), len(other_occs)))
    shuffle(not_contained)
    print(not_contained[:20])


def read_subprop_file(filepath) -> List[Tuple[Tuple[str, str], List[Tuple]]]:
    result: List[Tuple[Tuple[str, str], List[Tuple[str, str]]]] = []
    with open(filepath, 'r') as fin:
        for l in fin:
            ps = l.strip().split('\t')
            par_id, par_label = ps[0].split(',', 1)
            childs = [tuple(p.split(',')) for p in ps[1:]]
            result.append(((par_id, par_label), childs))
    return result


def read_nway_file(filepath,
                   filter_prop: set = None,
                   keep_one_per_prop: bool = False) -> List[Tuple[Tuple[str, Tuple], int]]:
    result = []
    seen_prop = set()
    with open(filepath, 'r') as fin:
        for l in fin:
            label, poccs = l.strip().split('\t')
            label = int(label)
            poccs = poccs.split(' ')
            assert len(poccs) % 2 == 1, 'nway file format error'
            occs = tuple(tuple(poccs[i * 2 + 1:i * 2 + 3]) for i in range((len(poccs) - 1) // 2))
            pid = poccs[0]
            if filter_prop and pid not in filter_prop:
                continue
            if keep_one_per_prop and pid in seen_prop:
                continue
            if keep_one_per_prop:
                seen_prop.add(pid)
            result.append(((pid, occs), label))
    return result


def read_multi_pointiwse_file(filepath,
                              filter_prop: set = None,
                              keep_one_per_prop: bool = False) \
        -> List[Tuple[Tuple[str, Tuple], Tuple[str, Tuple], int]]:
    result = []
    seen_prop = set()
    with open(filepath, 'r') as fin:
        for l in fin:
            label, p1o, p2o = l.strip().split('\t')
            label = int(label)
            p1 = p1o.split(' ')
            p2 = p2o.split(' ')
            assert len(p1) % 2 == 1 and len(p1) % 2 == 1, 'pointwise file format error'
            p1occs = tuple(tuple(p1[i * 2 + 1:i * 2 + 3]) for i in range((len(p1) - 1) // 2))
            p2occs = tuple(tuple(p2[i * 2 + 1:i * 2 + 3]) for i in range((len(p2) - 1) // 2))
            p1 = p1[0]
            p2 = p2[0]
            if p1 == p2:
                # pairs of two same properties should not be considered
                # TODO: this is just a workaround. An exception should be raised
                continue
            if filter_prop and (p1 not in filter_prop or p2 not in filter_prop):
                continue
            if keep_one_per_prop and (p1, p2) in seen_prop:
                continue
            if keep_one_per_prop:
                seen_prop.add((p1, p2))
                seen_prop.add((p2, p1))
            if len(set(p1occs) & set(p2occs)) > 0:
                # skip examples where two subgraphs overlap
                # TODO: this is just a workaround. An exception should be raised
                continue
            result.append(((p1, p1occs), (p2, p2occs), label))
    return result


def read_pointiwse_file(filepath,
                        filter_prop: set = None,
                        keep_one_per_prop: bool = False) \
        -> List[Tuple[Tuple[str, str, str], Tuple[str, str, str], int]]:
    print('deprecated')
    result = []
    seen_prop = set()
    with open(filepath, 'r') as fin:
        for l in fin:
            label, p1o, p2o = l.strip().split('\t')
            label = int(label)
            p1, h1, t1 = p1o.split(' ')
            p2, h2, t2 = p2o.split(' ')
            if filter_prop and (p1 not in filter_prop or p2 not in filter_prop):
                continue
            if keep_one_per_prop and (p1, p2) in seen_prop:
                continue
            if keep_one_per_prop:
                seen_prop.add((p1, p2))
                seen_prop.add((p2, p1))
            result.append(((h1, p1, t1), (h2, p2, t2), label))
    return result


def read_prop_file(filepath) -> List[str]:
    result = []
    with open(filepath, 'r') as fin:
        for l in fin:
            result.append(l.strip().split('\t')[0])
    return result


def read_subgraph_cache():
    def wrapper(func):
        @functools.wraps(func)
        def new_func(filepath, *args, **kwargs):
            cache_path = filepath + '.pickle'
            if os.path.exists(cache_path):
                print('load subgraph from cache ...')
                with open(cache_path, 'rb') as fin:
                    subgraph = pickle.load(fin)
            else:
                subgraph = func(filepath, *args, **kwargs)
                print('cache subgraph ...')
                with open(cache_path, 'wb') as fout:
                    pickle.dump(subgraph, fout)
            return subgraph
        return new_func
    return wrapper


def read_subgraph_file(filepath, only_root=False) -> Dict[str, List[Tuple[str, str, str]]]:
    print('load subgraphs ...')
    result = {}
    with open(filepath, 'r') as fin:
        for l in tqdm(fin):
            l = l.strip().split('\t', 1)
            root = l[0]
            if not only_root and len(l) > 1:
                adjs = [tuple(adj.split(' ')) for adj in l[1].split('\t')]
            else:
                adjs = []
            result[root] = adjs
    return result


def get_is_sibling(subprops: List[Tuple[Tuple[str, str], List[Tuple]]]):
    is_sibling = set()
    for p in subprops:
        for p1, p2 in combinations(p[1], 2):
            is_sibling.add((p1[0], p2[0]))
            is_sibling.add((p2[0], p1[0]))
    return is_sibling


def get_is_parent(subprops: List[Tuple[Tuple[str, str], List[Tuple]]]):
    is_parent = set()
    for prop in subprops:
        p = prop[0][0]  # parent pid
        for c in prop[1]:
            is_parent.add((p, c[0]))
    return is_parent


def filter_prop_occ_by_subgraph_and_emb(prop: str,
                                        prop_occs: Union[List[Tuple[str, str]], Iterable[Tuple[str, str]]],
                                        subgraph_dict: Dict,
                                        emb_set: set,
                                        max_num: int = None):
    ''' filter out property occurrence without subgraph or embedding '''
    filtered = []
    # emb check
    if prop not in emb_set:
        return filtered
    for occ in prop_occs:
        hid, tid = occ
        # subgraph check
        if hid not in subgraph_dict or tid not in subgraph_dict:
            continue
        # emb check
        try:
            if hid not in emb_set or tid not in emb_set:
                raise KeyError
            for two_side in [hid, tid]:
                for e1, p, e2 in subgraph_dict[two_side]:
                    if e1 not in emb_set or p not in emb_set or e2 not in emb_set:
                        raise KeyError
        except KeyError:
            continue
        filtered.append(occ)
        if max_num and len(filtered) >= max_num:
            break
    return filtered


class PropertySubtree():
    def __init__(self, tree: Tuple[str, List]):
        self.tree = tree


    @property
    def nodes(self):
        return list(self.traverse())


    @classmethod
    def build(cls, root: str, child_dict: Dict[str, List[str]]):
        subtree = get_subtree(root, child_dict)
        return cls(subtree)


    @staticmethod
    def traverse_subtree(subtree, ancestors=[], return_ancestors=False):
        if return_ancestors:
            yield subtree[0], ancestors
        else:
            yield subtree[0]
        na: List[str] = deepcopy(ancestors)
        na.append(subtree[0])
        for c in subtree[1]:
            yield from PropertySubtree.traverse_subtree(c, ancestors=na, return_ancestors=return_ancestors)


    def traverse(self, return_ancestors=False):
        yield from PropertySubtree.traverse_subtree(self.tree, ancestors=[], return_ancestors=return_ancestors)


    @staticmethod
    def split_within_subtree(subtree, tr, dev, te, return_parent: bool = False, filter_set: set = None):
        ''' split the subtree by spliting each tir into train/dev/test set '''
        parent = subtree[0]
        siblings = [c[0] for c in subtree[1] if filter_set is None or c[0] in filter_set]
        shuffle(siblings)
        trs = int(len(siblings) * tr)
        devs = int(len(siblings) * dev)
        tes = len(siblings) - tr - dev
        test_props = siblings[trs + devs:]
        dev_props = siblings[trs:trs + devs]
        train_props = siblings[:trs]
        if len(train_props) > 0 and len(dev_props) > 0 and len(test_props) > 0 and \
                (filter_set is None or parent in filter_set):
            if return_parent:
                yield parent, train_props, dev_props, test_props
            else:
                yield train_props, dev_props, test_props
        for c in subtree[1]:
            yield from PropertySubtree.split_within_subtree(
                c, tr, dev, te, return_parent=return_parent, filter_set=filter_set)


    def split_within(self, tr, dev, te, return_parent: bool = False, filter_set: set = None):
        yield from PropertySubtree.split_within_subtree(
            self.tree, tr, dev, te, return_parent=return_parent, filter_set=filter_set)


    @staticmethod
    def get_depth_subtree(subtree: Tuple[str, List]) -> int:
        depth = 1
        max_depth = 0
        for c in subtree[1]:
            d = PropertySubtree.get_depth_subtree(c)
            if d > max_depth:
                max_depth = d
        return depth + max_depth


    def get_depth(self) -> int:
        return PropertySubtree.get_depth_subtree(self.tree)


    @staticmethod
    def print_subtree(subtree: Tuple[str, List],
                      id2label: Dict[str, str] = None,
                      defalut_label: str = '',
                      prefix: str = '') -> List[str]:
        id = subtree[0]
        label = id2label[id] if id2label and id in id2label else defalut_label
        l = prefix + id + ': ' + label
        ls = [l]
        for c in subtree[1]:
            ls.extend(PropertySubtree.print_subtree(
                c, id2label=id2label, defalut_label=defalut_label, prefix=prefix + '\t'))
        return ls


    def print(self,
              id2label: Dict[str, str] = None,
              defalut_label: str = '',
              prefix: str = '') -> str:
        return '\n'.join(PropertySubtree.print_subtree(
            self.tree, id2label=id2label, defalut_label=defalut_label, prefix=prefix))


def get_subtree(root: str, child_dict: Dict[str, List[str]]) -> Tuple[str, List[Tuple[str, List]]]:
    if root not in child_dict:
        return (root, [])
    result = (root, [get_subtree(c, child_dict) for c in child_dict[root]])
    return result


def get_all_subtree(subprops: List[Tuple[Tuple[str, str], List[Tuple]]]) \
        -> Tuple[List[PropertySubtree], List[PropertySubtree]]:
    num_prop = len(subprops)
    print('{} props'.format(num_prop))

    # get parent link and children link
    parent_dict = defaultdict(lambda: [])
    child_dict = defaultdict(lambda: [])
    for p in subprops:
        parent_id = p[0][0]
        child_dict[parent_id] = [c[0] for c in p[1]]
        for c in p[1]:
            parent_dict[c[0]].append(parent_id)

    # construct tree for properties without parent
    subtrees: List[PropertySubtree] = []
    isolate: List[PropertySubtree] = []
    for p in subprops:
        pid = p[0][0]
        if len(parent_dict[pid]) == 0:
            subtree = PropertySubtree.build(pid, child_dict)
            if subtree.get_depth() > 1:
                subtrees.append(subtree)
            else:
                isolate.append(subtree)

    print('{} subtree'.format(len(subtrees)))
    print('avg depth: {}'.format(np.mean([s.get_depth() for s in subtrees])))
    print('{} isolated prop'.format(len(isolate)))

    return subtrees, isolate


def get_is_ancestor(subtrees: List[PropertySubtree]):
    is_ancestor = set()
    for subtree in subtrees:
        for child, ancestors in subtree.traverse(return_ancestors=True):
            for anc in ancestors:
                is_ancestor.add((anc, child))
    return is_ancestor


class PropertyOccurrence():
    def __init__(self,
                 pid2occs: Dict[str, List[Tuple]],
                 num_occ_per_subgraph: int = 1):
        self.pid2occs = pid2occs
        self.num_occ_per_subgraph = num_occ_per_subgraph
        self._pid2multioccs = dict((pid, self.group_occs(pid, num_occ_per_subgraph)) for pid in pid2occs)


    @classmethod
    def build(cls,
              pids: List[str],
              prop_occ_dir: str,
              subgraph_dict: dict = None,
              emb_set: set = None,
              max_occ_per_prop: int = None,
              min_occ_per_prop: int = None,  # property with number of occ less than this will be removed
              num_occ_per_subgraph: int = 1,
              populate_method: str = None,
              subtrees: List[PropertySubtree] = None):
        pid2occs: Dict[str, List[Tuple]] = {}
        num_long_tail_prop = 0
        for p in tqdm(pids):
            occs = read_prop_occ_file_from_dir(
                p, prop_occ_dir, filter=True, contain_name=False, use_order=True)
            if subgraph_dict is not None and emb_set is not None:
                occs = filter_prop_occ_by_subgraph_and_emb(
                    p, occs, subgraph_dict, emb_set, max_num=max_occ_per_prop)  # check existence
            else:
                new_occs = []
                for i, occ in enumerate(occs):
                    if i >= max_occ_per_prop:
                        break
                    new_occs.append(occ)
                occs = new_occs
            if len(occs) == 0:
                continue  # skip empty property
            if min_occ_per_prop is not None and len(occs) < min_occ_per_prop:
                num_long_tail_prop += 1
                continue  # skip long tail properties
            shuffle(occs)
            if max_occ_per_prop:
                occs = occs[:max_occ_per_prop]
            # TODO: number of occs of different properties are unbalanced
            pid2occs[p] = occs
        if min_occ_per_prop is not None:
            print('remove {} long tail properties with threshold {}'.format(
                num_long_tail_prop, min_occ_per_prop))
        if populate_method is not None:
            print('populate properties recursively')
            new_pid2occs: Dict[str, set] = {}
            for subtree in tqdm(subtrees):
                PropertyOccurrence.recursive_complete(
                    pid2occs, subtree.tree, new_pid2occs, populate_method=populate_method)
            pid2occs = dict((k, list(v)) for k, v in new_pid2occs.items())
        return cls(pid2occs, num_occ_per_subgraph=num_occ_per_subgraph)


    @staticmethod
    def recursive_complete(pid2occs: Dict[str, List[Tuple]],
                           subtree: Tuple[str, List],
                           new_pid2occs: Dict[str, set],
                           populate_method: str = 'combine_child'):
        # in all these populate_method, overlap should be avoided
        # 'combine_child': occs of a property is the union of all its children
        # 'combine_child_and_self': occs of a property is the union of all its children and itself
        parent, childs = subtree
        if parent in new_pid2occs:  # has been processed
            return
        if len(childs) == 0:  # leaf node
            if parent in pid2occs:
                new_pid2occs[parent] = set(pid2occs[parent])
            return
        for child in childs:  # none-leaf node, go deeper
            PropertyOccurrence.recursive_complete(
                pid2occs, child, new_pid2occs, populate_method=populate_method)
        # populate current property using children
        if populate_method in {'combine_child', 'combine_child_and_self'}:
            # collect all occs of children
            childs_occs = set()
            for child in childs:
                if child[0] not in new_pid2occs:
                    continue
                occs = new_pid2occs[child[0]]
                # TODO: node with multiple parents are used multiple times
                childs_occs.update(occs)
            # give first half to parent
            childs_occs = list(childs_occs)
            shuffle(childs_occs)
            parent_occs = set(childs_occs[:len(childs_occs) // 3])  # TODO: how to set up the number
            # remove them from children
            for child in childs:
                if child[0] not in new_pid2occs:
                    continue
                new_pid2occs[child[0]] -= parent_occs  # parent's parent can still overlap with these children
            if populate_method == 'combine_child_and_self' and parent in pid2occs:
                parent_occs.update(pid2occs[parent])  # add self
            if len(parent_occs) > 0:
                new_pid2occs[parent] = parent_occs
        else:
            raise NotImplementedError


    @property
    def pids(self) -> List[str]:
        return list(self.pid2occs.keys())


    def __getitem__(self, item: str) -> List[Tuple]:
        return self.pid2occs[item]


    def group_occs(self, pid, size) -> List[List[Tuple]]:
        occs = self.pid2occs[pid]
        return [occs[i:i+size] for i in range(0, len(occs), size)]


    def get_all_occs(self, pid: str, num_sample: int) -> Iterable[List]:
        occs = self._pid2multioccs[pid]
        for i in range(min(num_sample, len(occs))):
            yield occs[i]


    def get_all_pairs(self,
                      pid1: str,
                      pid2: str,
                      num_sample: int,
                      sam_for_pid1: int = 1,
                      sam_for_pid2: int = 1) -> Iterable[Tuple[List, List]]:
        p1occs = self._pid2multioccs[pid1]
        p2occs = self._pid2multioccs[pid2]

        sam_prob = min(1, num_sample / (len(p1occs) * len(p2occs)))

        if sam_prob == 1 and sam_for_pid1 == 1 and sam_for_pid2 == 1:
            for p1o in p1occs:
                for p2o in p2occs:
                    yield p1o, p2o  # p1o and p2o are lists of occurrences
        else:
            for i in range(num_sample):
                p1o = self.sample_n_occs(pid1, sam_for_pid1)
                p2o = self.sample_n_occs(pid2, sam_for_pid2)
                yield p1o, p2o  # p1o and p2o are lists of occurrences


    def sample_n_occs(self, pid: str, n: int = 1):
        occs: List[Tuple] = []
        poccs = self._pid2multioccs[pid]
        while n > 0:
            occs.extend(poccs[min(int(len(poccs) * random.random()), len(poccs) - 1)])
            n -= 1
        return occs
