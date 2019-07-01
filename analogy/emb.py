from typing import Dict, List, Any
import random
import os
import numpy as np
from tqdm import tqdm
import pickle
import multiprocessing
import time
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import multivariate_normal
from sklearn.neighbors.kde import KernelDensity
from wikiutil.util import load_tsv_as_dict
from wikiutil.property import read_subprop_file, get_is_parent, get_is_ancestor, get_pid2plabel, get_all_subtree


def get_rel(parent, child, is_parent, is_ancestor):
    if (parent, child) in is_parent:
        return 'parent'
    elif (parent, child) in is_ancestor:
        return 'ancestor'
    elif child.startswith(parent + '_'):
        return 'parent'
    else:
        return '*'


def get_label_dist(props, labels, is_parent, pid2plabel):
    dist = []
    for p in props:
        for l in labels:
            if (l, p) in is_parent:
                dist.append(pid2plabel[l])
    u, c = np.unique(dist, return_counts=True)
    print(sorted(zip(u, c), key=lambda x: -x[1]))


def sim_func(child_emb, parent_emb, method='cosine', sigma=1.0, kde_c=None, kde_p=None):
    if method == 'cosine':
        return np.mean(cosine_similarity(child_emb, parent_emb))

    if method == 'expeuc':
        dist = np.expand_dims(child_emb, 1) - np.expand_dims(parent_emb, 0)
        dist = np.sum(dist * dist, -1)
        dist_min = np.min(dist, -1)
        sim = np.exp(-dist + np.expand_dims(dist_min, -1))
        sim = np.mean(np.log(np.mean(sim, -1)) - dist_min)
        return sim

    if method == 'mixgau':
        emb_dim = child_emb.shape[1]
        num_occ_child = child_emb.shape[0]
        num_occ_parent = parent_emb.shape[0]
        rv = multivariate_normal(np.zeros(emb_dim), np.diag(np.ones(emb_dim)) * sigma)
        dist = np.expand_dims(child_emb, 1) - np.expand_dims(parent_emb, 0)
        sim = rv.pdf(dist)
        sim = np.sum(np.log(np.sum(sim / num_occ_parent, axis=-1)))
        return sim

    if method == 'mixgau_fast':
        emb_dim = child_emb.shape[1]
        num_occ_child = child_emb.shape[0]
        num_occ_parent = parent_emb.shape[0]
        var = np.array([sigma] * emb_dim)
        # SHAPE: (num_occ_child, num_occ_parent, emb_dim)
        dist = np.expand_dims(child_emb, 1) - np.expand_dims(parent_emb, 0)
        dist = np.reshape(dist, (-1, emb_dim))
        denominator = -0.5 * (emb_dim * (np.log(2 * np.pi)) + np.sum(np.log(var)))
        numerator = -0.5 * np.sum(dist ** 2 / np.expand_dims(var, 0), -1)
        log_prob = np.reshape(denominator + numerator, (num_occ_child, num_occ_parent))
        log_prob_max = np.max(log_prob, -1, keepdims=True)
        sim = np.sum(np.log(np.sum(np.exp(log_prob - log_prob_max) / num_occ_parent, axis=-1)) + log_prob_max)
        return sim

    if method == 'avg':
        child_emb = np.mean(child_emb, 0)
        parent_emb = np.mean(parent_emb, 0)
        return np.mean(cosine_similarity([child_emb], [parent_emb]))

    if method == 'kde':
        if kde_c is None:
            kde_c = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(child_emb)
        if kde_p is None:
            kde_p = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(parent_emb)
        child_prob = kde_c.score_samples(child_emb)
        parent_prob = kde_p.score_samples(child_emb)
        kl = np.sum(np.exp(child_prob) * (child_prob - parent_prob))
        return -kl

    raise NotImplementedError


def compute_one(this_pid,
                label_embs=None, detect_cheat=None, is_parent=None,
                is_ancestor=None, method=None, **kwargs):
    pid, poccs, pid_emb = this_pid
    ranks = []
    for l, ls, ls_emb in label_embs:
        if pid == l:
            continue
        if detect_cheat and (l, pid) in is_ancestor and len(set(poccs) & set(ls)) > 0:
            raise Exception('parent {} child {} is cheating'.format(pid, l))
        sim = sim_func(pid_emb, ls_emb, method=method, **kwargs)
        ranks.append((l, sim))

    ranks = sorted(ranks, key=lambda x: -x[1])
    ranks = [(l, sim, get_rel(l, pid, is_parent, is_ancestor)) for l, sim in ranks]
    return pid, ranks


def compute_overlap(data_dir, split, poccs, subprops, emb, emb_id2ind, top=100, method='cosine',
                    only_prop_emb=False, detect_cheat=True, use_minus=False,
                    filter_num_poccs=None, filter_pids=None, num_workers=1, skip_split=False,
                    ori_subprops=None, debug=False, **kwargs):
    emb_dim = emb.shape[1]
    print('#words in emb: {}'.format(len(emb_id2ind)))

    # load poccs
    poccs = os.path.join(data_dir, poccs)
    with open(poccs, 'rb') as fin:
        poccs = pickle.load(fin)

    # sample occurrences for each property
    all_ids = set()
    sampled_poccs = {}
    for k in poccs:
        if top:
            sampled_poccs[k] = random.sample(poccs[k], len(poccs[k]))[:top]
        else:
            sampled_poccs[k] = poccs[k]
        if len(sampled_poccs[k]) > 0:
            heads, tails = zip(*sampled_poccs[k])
            all_ids.update(heads)
            all_ids.update(tails)
    all_ids_emb = all_ids & set(emb_id2ind.keys())
    print('#entities in property occs: {}, {} have embs'.format(len(all_ids), len(all_ids_emb)))

    # load properties for test
    if type(split) is str:
        split = [split]
    props = []
    for sp in split:
        props += list(load_tsv_as_dict(os.path.join(data_dir, sp)).keys())

    # load labels/parents
    labels = list(load_tsv_as_dict(os.path.join(data_dir, 'label2ind.txt')).keys())

    # load subprops
    subprops = read_subprop_file(subprops)
    is_parent = get_is_parent(subprops)
    subtrees, _ = get_all_subtree(subprops)
    is_ancestor = get_is_ancestor(subtrees)
    pid2plabel = get_pid2plabel(subprops)
    if skip_split:
        ori_subprops = read_subprop_file(ori_subprops)
        ori_parents = set([p for p, c in get_is_parent(ori_subprops)])

    print('#property: {} #label: {}'.format(len(props), len(labels)))
    if debug:
        get_label_dist(props, labels, is_parent, pid2plabel)

    # collect embs for labels
    label_embs = []
    for l in labels:
        if l not in sampled_poccs or len(sampled_poccs[l]) <= 0:
            continue
        if skip_split and l not in ori_parents:
            continue
        ls = sampled_poccs[l]
        if filter_num_poccs and len(poccs[l]) < filter_num_poccs:
            continue
        if only_prop_emb:
            ls_emb = np.array([emb[emb_id2ind[l]]])
        else:
            ls_emb = [emb[emb_id2ind[e]] for o in ls for e in o if o[0] in emb_id2ind and o[1] in emb_id2ind]
            ls_emb = np.array(ls_emb).reshape(-1, 2 * emb_dim)
        if use_minus:
            ls_emb = ls_emb[:, :emb_dim] - ls_emb[:, emb_dim:]
        if ls_emb.shape[0] <= 0:
            continue
        label_embs.append((l, ls, ls_emb))

    # collect embs for properties
    prop_embs = []
    for tp in props:
        if filter_pids and tp not in filter_pids:
            continue
        has_parent = False
        for l, _, _ in label_embs:
            if (l, tp) in is_parent:
                has_parent = True
        if not has_parent:
            continue
        tps = sampled_poccs[tp]
        if filter_num_poccs and len(poccs[tp]) < filter_num_poccs:
            continue
        if only_prop_emb:
            tps_emb = np.array([emb[emb_id2ind[tp]]])
        else:
            tps_emb = [emb[emb_id2ind[e]] for o in tps for e in o if o[0] in emb_id2ind and o[1] in emb_id2ind]
            tps_emb = np.array(tps_emb).reshape(-1, 2 * emb_dim)
        if use_minus:
            tps_emb = tps_emb[:, :emb_dim] - tps_emb[:, emb_dim:]
        if len(tps_emb) == 0:
            continue
        prop_embs.append((tp, tps, tps_emb))

    # iterate over properties
    start_time = time.time()
    pool = multiprocessing.Pool(num_workers)
    results = pool.map(partial(
        compute_one, label_embs=label_embs, detect_cheat=detect_cheat, is_parent=is_parent,
        is_ancestor=is_ancestor, method=method, **kwargs), prop_embs)
    print('use {} secs'.format(time.time() - start_time))

    # collect results
    total, correct, mrr = 0, 0, 0
    correct_li = []
    rank_dict: Dict[str, List] = {}
    kde_dict: Dict[str, Any] = {}
    for result in results:
        if result == None:
            continue
        tp, rank = result
        rank_dict[tp] = rank
        if (rank[0][0], tp) in is_parent:
            correct += 1
            correct_li.append((tp))
        rr = 0
        for i in range(len(rank)):
            if (rank[i][0], tp) in is_parent:
                rr = 1 / (i + 1)
                break
        mrr += rr
        total += 1

    print('acc {}, mrr {}, total {} properties and {} labels'.format(
        correct / total, mrr / total, total, len(label_embs)))
    if debug:
        get_label_dist(correct_li, labels, is_parent, pid2plabel)
        print('correct list:')
        print(correct_li)

    return rank_dict
