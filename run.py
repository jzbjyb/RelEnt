#!/usr/bin/env python


import os, argparse, shutil, json
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, fcluster
from urllib.parse import unquote
from wikiutil.util import load_shyamupa_t2id, file_filter_by_key, plot_correlation


def load_shyamupa_count(filepath):
    d = {}
    with open(filepath, 'r') as fin:
        for l in fin:
            l = l.strip()
            if len(l) == 0:
                continue
            k, v, c = l.split('\t')
            c = int(c)
            d[k] = c
    return d


def load_shuyan_ptitle_pid_map(filepath):
    d = {}
    ec = 0
    tc = 0
    with open(filepath, 'r') as fin:
        for l in fin:
            l = l.strip()
            if len(l) == 0:
                continue
            tc += 1
            try:
                k, v = l.split(' ||| ')
            except:
                ec += 1
                continue
            k = k.replace(' ', '_')
            d[k] = v
    print('loaded ptitle_pid_map, err/total {}/{}'.format(ec, tc))
    return d


def load_shuyan_pid_prior_map(filepath):
    d = {}
    with open(filepath, 'r') as fin:
        for l in fin:
            l = l.strip()
            if len(l) == 0:
                continue
            k, v = l.split(' ||| ')
            v = float(v)
            d[k] = v
    return d


@file_filter_by_key(cache_dir='cache')
def load_pid_cate_map(filepath, filterkeys=None):
    if filterkeys is None:
        d = {}
        with open(filepath, 'r') as fin:
            for l in fin:
                l = l.strip()
                if len(l) == 0:
                    continue
                k, v = l.split('\t')
                v = v.split(' ')
                d[k] = v
        return d
    else:
        def gen():
            with open(filepath, 'r') as fin:
                for l in fin:
                    l = l.strip()
                    if len(l) == 0:
                        continue
                    k, v = l.split('\t')
                    if k in filterkeys:
                        yield l
        return gen


def load_cateid_cate_map(filepath):
    d = {}
    with open(filepath, 'r') as fin:
        for l in fin:
            l = l.strip()
            if len(l) == 0:
                continue
            k, v = l.split('\t')
            d[k] = v
    return d


def iter_wiki_json_file(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            yield os.path.join(root, file)


def sample_wiki_json_files(root_dir, num_sam=10):
    '''
    Smaple `num_sam` files from the `root_dir` randomly and concatenate them.
    '''
    allfs = [f for f in iter_wiki_json_file(root_dir)]
    num_sam = min(len(allfs), num_sam)
    print('sample {} from totally {} wiki json files'.format(num_sam, len(allfs)))
    samfs = np.random.permutation(allfs)[:num_sam]
    return samfs


def buld_puri_euri_mapping(wiki_file, url_prefix='http://en.wikipedia.org/wiki/'):
    '''
    Convert a wiki json file to two mappings (page2entity and entity2page) and frequency dict of entities.
    '''
    count = defaultdict(lambda: 0)
    p2e = defaultdict(lambda: [])
    e2p = defaultdict(lambda: [])

    if url_prefix is not None:
        print('use {} as url prefix'.format(url_prefix))
        upl = len(url_prefix)
    else:
        upl = None
        print('WARN: use slash to find uri in url will make mistakens when the title has slash')

    with open(wiki_file, 'r') as fin:
        for l in fin:
            l = l.strip()
            if len(l) == 0:
                continue
            d = json.loads(l)
            if upl is None:
                puri = d['url'].rsplit('/', 1)[-1]
            else:
                puri = d['url'][upl:]
            # remember to unquote
            puri = unquote(puri)
            # some uri are null pointer
            euris = [unquote(a['uri']) for a in d['annotations']]
            # bi-directional mapping
            p2e[puri].extend(euris)
            p2e[puri].append(puri)
            for euri in euris:
                e2p[euri].append(puri)
            e2p[puri].append(puri)
            # entity count
            for uri in [puri] + euris:
                count[uri] += 1
    return p2e, e2p, count


def split_by_freq(wiki_file, ratio=0.5, pt2pid=None, pid2prior=None):
    '''
    Split `wiki_file` into a general-purpose and a domain-specific KB by entity frequency.
    '''
    # get statistics
    p2e, e2p, count = buld_puri_euri_mapping(wiki_file)
    # use dataset-wide statistics
    if pt2pid is not None and pid2prior is not None:
        oov = 0
        for k, v in count.items():
            if k in pt2pid and pt2pid[k] in pid2prior:
                count[k] = pid2prior[pt2pid[k]]
            else:
                oov += 1
                count[k] = 0
        print('{} out of {} are oov'.format(oov, len(count)))
    count_sort = sorted(count.items(), key=lambda x: -x[1])
    plt.plot(np.log(list(range(len(count_sort)))), np.log([e[1] for e in count_sort]))
    plt.savefig('stat.png')
    print('#page {}, #entity {}'.format(len(p2e), len(e2p)))

    # get top entities
    csum = np.sum([e[1] for e in count_sort])
    cthres = csum * ratio
    print('total occurrence is {} and thres hold is {}'.format(csum, cthres))
    c, topk = 0, 0
    for i in range(len(count_sort)):
        c += count_sort[i][1]
        if c >= cthres:
            topk = i
            break
    entity_li = list([e[0] for e in count_sort[:topk]])
    page_set = set([p for e in entity_li for p in e2p[e]])
    print('get top {} entities as initial seeds (occur in {} pages)'.format(topk, len(page_set)))
    print(count_sort[:topk][:10])

    # recursively expansion
    print('recursively include entities that co-occur with top entites')
    page_set = set()
    entity_set = set(entity_li)
    i = 0
    while i < len(entity_set):
        entity = entity_li[i]
        for page in e2p[entity]:
            if page not in page_set:
                page_set.add(page)
                for ne in p2e[page]:
                    if ne not in entity_set:
                        entity_set.add(ne)
                        entity_li.append(ne)
        i += 1

    # get general-purse and domain KBs
    rm_entity_set = set(e2p.keys()) - entity_set
    rm_page_set = set(p2e.keys()) - page_set
    print('general KB: page {}, #entity {}'.format(len(page_set), len(entity_set)))
    print('domain KB: page {}, #entity {}'.format(len(rm_page_set), len(rm_entity_set)))


def get_all_cateid(leaf_cate, cateid2cateid, root={'14105005', '7345184'}):
    '''
    Recursively get all the categories starting from the leaf categories from `leaf_cate`
    `root` is the id of the root category, whose default value are `Category:Content`
    and `Category:Main_topic_classifications`
    '''
    all = set()
    def recur(cate_set, all):
        if len(cate_set) == 0:
            return
        next_cate_set = set()
        all |= cate_set
        reach_root = False
        for cate in cate_set:
            if cate in cateid2cateid: # has parent cates
                ne = set(cateid2cateid[cate])
                if len(ne & root) > 0:
                    reach_root = True
                    break
                next_cate_set |= ne
        if not reach_root: # stop when reaching root, otherwise categories become unstable
            next_cate_set -= all
            recur(next_cate_set, all)
    recur(leaf_cate, all)
    return all


def split_by_cate(wiki_file, corr_fig, pt2pid, pid2cateid, cateid2cateid, restrict_cateid2cate):
    '''
    Split `wiki_file` into a general-purpose and a domain-specific KB by category.
    '''
    # get statistics
    p2e, e2p, count = buld_puri_euri_mapping(wiki_file)
    # get category count
    cate_count = defaultdict(lambda: 0)
    oov = 0
    for p in p2e:
        if p in pt2pid and pt2pid[p] in pid2cateid:
            for c in pid2cateid[pt2pid[p]]:
                cate_count[c] += 1
        else:
            oov += 1
    print('#page {}, #entity {}, #cate {} with {} oov'.format(
        len(p2e), len(e2p), len(cate_count), oov))
    cate_count_sort = sorted(cate_count.items(), key=lambda x: -x[1])
    print(cate_count_sort[:20])
    # get restrict category co-occurrence
    cate_co = defaultdict(lambda: set())
    all_es = set()
    for i, p in enumerate(p2e):
        es = set(p2e[p])
        all_es |= es
        if p in pt2pid and pt2pid[p] in pid2cateid:
            all_cate = get_all_cateid(set(pid2cateid[pt2pid[p]]), cateid2cateid)
            for c in all_cate:
                if c not in restrict_cateid2cate:
                    continue
                cate_co[c] |= es
    pmi = np.zeros((len(cate_co), len(cate_co)))
    cate_co = cate_co.items()
    for i, (c1, e1) in enumerate(cate_co):
        for j, (c2, e2) in enumerate(cate_co):
            pmi[i, j] = np.log((len(all_es) * len(e1 & e2) / (len(e1) * len(e2))) or 1e-5)

    pmi_cate = np.array([restrict_cateid2cate[c[0]] for c in cate_co])
    # first rank by count
    rank = np.argsort([-len(c[1]) for c in cate_co])
    pmi_cate = pmi_cate[rank]
    pmi = pmi[:, rank][rank]
    # cluster
    pdist = np.array([pmi[i, j] for i in range(len(pmi)) for j in range(len(pmi)) if i < j])
    pdist = np.max(pdist) - pdist
    clu = fcluster(ward(pdist), t=2, criterion='distance')
    print('clusters: {}'.format(clu))
    rank = np.argsort(clu)
    pmi_cate = pmi_cate[rank]
    pmi = pmi[:, rank][rank]
    print(pmi_cate)
    with np.printoptions(precision=3):
        print(pmi)
    plot_correlation(pmi, pmi_cate, 'category correlation', corr_fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str,
                        choices=['sample', 'split-freq', 'split-cate'], required=True)
    parser.add_argument('-data', type=str, required=True)
    parser.add_argument('-out', type=str, default=None)
    parser.add_argument('-eprior', help='path to the entity prior file', type=str, default=None)
    parser.add_argument('-pt2pid', help='path to the mapping file from wiki title to wiki page id',
                        type=str, default=None)
    parser.add_argument('-pid2cate', help='path to the mapping file from page id to category id',
                        type=str, default=None)
    args = parser.parse_args()

    if args.task == 'sample':
        fs = sample_wiki_json_files(args.data, num_sam=50)
        # save the files from which we sampled
        with open(args.out + '.from', 'w') as fout:
            for f in fs:
                fout.write('{}\n'.format(f))
        # combine wiki files
        with open(args.out, 'wb') as fout:
            for f in fs:
                with open(f, 'rb') as fin:
                    shutil.copyfileobj(fin, fout, 1024 * 1024 * 10)
    elif args.task.startswith('split-freq'):
        # more comprehensive
        _, e2p, _ = buld_puri_euri_mapping(args.data)
        pt2pid = load_shyamupa_t2id('data/enwiki-20181020.id2t', args.data, set(e2p.keys()))
        pid2prior = load_shyamupa_count('data/enwiki-20181020.counts')
        # less comprehensive
        #pt2pid = load_shuyan_ptitle_pid_map('data/title_id_map')
        #pid2prior = load_shuyan_pid_prior_map('data/eprior_normalize')
        split_by_freq(args.data, ratio=0.2, pt2pid=pt2pid, pid2prior=pid2prior)
    elif args.task.startswith('split-cate'):
        _, e2p, _ = buld_puri_euri_mapping(args.data)
        pt2pid = load_shyamupa_t2id('data/enwiki-20181020.id2t', args.data, set(e2p.keys()))
        pid2cateid = load_pid_cate_map('result/pid2cateid.tsv', args.data,
                                       set(pt2pid[k] for k in e2p.keys() if k in pt2pid))
        cateid2cateid = load_pid_cate_map('result/cateid2cateid.tsv', None)
        cate = load_cateid_cate_map('result/wiki_main_cate_id.tsv')
        print('restricted cate: {}'.format(cate))
        split_by_cate(args.data, 'corr_test.png', pt2pid=pt2pid, pid2cateid=pid2cateid,
                      cateid2cateid=cateid2cateid, restrict_cateid2cate=cate)
