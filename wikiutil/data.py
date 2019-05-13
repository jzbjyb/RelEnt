from typing import Tuple, List, Any, Dict
from collections import defaultdict, namedtuple
from random import shuffle
from tqdm import tqdm
import numpy as np
from .property import read_pointiwse_file, read_subgraph_file


class DataPrepError(Exception):
    pass


def save_emb_ids(filepath, out_filepath):
    result = set()
    with open(filepath, 'r') as fin, open(out_filepath, 'w') as fout:
        for l in tqdm(fin):
            l = l.split('\t', 1)
            id = l[0]
            result.add(id)
            fout.write(id + '\n')


def read_emb_ids(filepath, filter=False) -> set:
    print('load emb ids ...')
    result = set()
    with open(filepath, 'r') as fin:
        for id in tqdm(fin):
            id = id.strip()
            if filter:
                if id.startswith('<http://www.wikidata.org/entity/') or \
                        id.startswith('<http://www.wikidata.org/prop/direct/'):
                    id = id.rsplit('/', 1)[1][:-1]
                    result.add(id)
            else:
                result.add(id)
    return result


def load_embedding(filepath) -> Dict[str, List[float]]:
    print('load emb ...')
    result = {}
    with open(filepath, 'r') as fin:
        for l in tqdm(fin):
            l = l.split('\t')
            id = l[0]
            emb = list(map(float, l[1:]))
            result[id] = emb
    return result


def filer_embedding(in_filepath: str,
                    out_filepath: str,
                    ids: set):
    print('filter emb ...')
    with open(in_filepath, 'r') as fin, open(out_filepath, 'w') as fout:
        num_entity, num_prop, dim = fin.readline().split('\t')
        for i, l in tqdm(enumerate(fin)):
            ls = l.split('\t', 1)
            id = ls[0]
            if id.startswith('<http://www.wikidata.org/entity/') or \
                    id.startswith('<http://www.wikidata.org/prop/direct/'):
                id = id.rsplit('/', 1)[1][:-1]
            if id not in ids:
                continue
            fout.write(id + '\t' + ls[1])


def filter_prop_occ_by_subgraph_and_emb(prop: str,
                                        prop_occs: List[Tuple[str, str]],
                                        subgraph_dict: Dict,
                                        emb_set: set):
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
    return filtered


class PropertySubgraph():
    def __init__(self,
                 property_id: str,
                 head_id: str,
                 tail_id: str,
                 subgraph_dict: Dict[str, List[Tuple[str, str, str]]],
                 emb_dict: Dict[str, List[float]] = None,
                 emb_size: int = None,
                 edge_type: str = 'one'):
        self.pid = property_id
        self.hid = head_id
        self.tid = tail_id
        try:
            self.hsg: List[Tuple[str, str, str]] = subgraph_dict[head_id]
            self.tsg: List[Tuple[str, str, str]] = subgraph_dict[tail_id]
        except KeyError:
            raise DataPrepError
        assert edge_type in {'one'}
        self.edge_type = edge_type
        self.id2ind, self.adjs, self.emb = self._to_gnn_input(emb_dict, emb_size)


    def _to_gnn_input(self,
                     emb_dict: Dict[str, List[float]] = None,
                     emb_size: int = None):
        ''' convert to gnn input format (adjacency list and input emb) '''

        if self.edge_type == 'one':
            # build adj list
            id2ind = defaultdict(lambda: len(id2ind))  # entity/proerty id to index mapping
            _ = id2ind[self.pid]  # make sure that the central property is alway indexed as 0
            adjs = []
            adjs.append((id2ind[self.hid], id2ind[self.pid]))
            adjs.append((id2ind[self.pid], id2ind[self.tid]))
            for two_side in [self.hsg, self.tsg]:
                for e1, p, e2 in two_side:
                    e1 = id2ind[e1]
                    p = id2ind[p]
                    e2 = id2ind[e2]
                    adjs.append((e1, p))
                    adjs.append((p, e2))

            # build emb
            ind2id = dict((v, k) for k, v in id2ind.items())
            if emb_dict is not None:
                try:
                    emb = np.array([emb_dict[ind2id[i]] for i in range(len(id2ind))])
                except KeyError:
                    raise DataPrepError
            else:
                emb = np.random.normal(0, 0.1, (len(id2ind), emb_size))

            return id2ind, adjs, emb



class PointwiseDataLoader():
    def __init__(self,
                 train_file: str,
                 dev_file: str,
                 test_file: str,
                 subgraph_file: str,
                 emb_size: int = None,
                 emb_file: str = None,
                 edge_type: str = 'one'):
        print('load data ...')
        self.train_list = read_pointiwse_file(train_file)
        self.train_list = self.pos_neg_filter(self.train_list, neg_ratio=5)  # TODO: add neg_ratio param
        self.dev_list = read_pointiwse_file(dev_file)
        self.test_list = read_pointiwse_file(test_file)
        self.subgraph_dict = read_subgraph_file(subgraph_file)
        assert edge_type in {'one'}
        # 'one' means only use a single type of edge to link properties and entities
        self.edge_type = edge_type
        self.emb_size = emb_size
        self.emb_dict = load_embedding(emb_file) if emb_file else None
        print('prep data ...')
        self.all_ids = set()  # all the entity ids and property ids used
        self._subgraph_cache: Dict[Tuple[str, str, str], PropertySubgraph] = {}  # cache subgraphs
        self.train_graph = self.create_pointwise_samples(self.train_list)
        print('train {} -> {}'.format(len(self.train_list), len(self.train_graph)))
        self.dev_graph = self.create_pointwise_samples(self.dev_list)
        print('dev {} -> {}'.format(len(self.dev_list), len(self.dev_graph)))
        self.test_graph = self.create_pointwise_samples(self.test_list)
        print('test {} -> {}'.format(len(self.test_list), len(self.test_graph)))


    def pos_neg_filter(self, samples: List[Tuple[Any, Any, int]], neg_ratio: int = 3):
        # used to filter negative samples for pointwise methods
        pos = [s for s in samples if s[2] > 0]
        neg = [s for s in samples if s[2] == 0]
        shuffle(neg)
        return pos + neg[:len(pos) * neg_ratio]


    def create_pointwise_samples(self, data: List):
        result = []
        for sample in tqdm(data):
            p1o, p2o, label = sample
            sample = (self.create_subgraph(p1o), self.create_subgraph(p2o), label)
            result.append(sample)
        return result


    def create_subgraph(self, property_occ: Tuple[str, str, str]) -> PropertySubgraph:
        hid, pid, tid = property_occ

        if property_occ not in self._subgraph_cache:
            self._subgraph_cache[property_occ] = PropertySubgraph(
                pid, hid, tid, self.subgraph_dict, self.emb_dict, self.emb_size, self.edge_type)

        sg = self._subgraph_cache[property_occ]
        self.all_ids |= sg.id2ind.keys()

        return sg


    def renew_data_iter(self, split='train'):
        data = getattr(self, split + '_graph')
        def iter(split, data):
            if split == 'train':  # shuffle train data
                perm = np.random.permutation(len(data))
            else:
                perm = range(len(data))
            for i in perm:
                yield data[i]
        setattr(self, split + '_iter', iter(split, data))


    def get_data_iter(self, split='train'):
        name = split + '_iter'
        if not hasattr(self, name):
            self.renew_data_iter(split)
        return getattr(self, name)


    def batch_iter(self,
                   split='train',
                   batch_size=64,
                   batch_per_epoch: int = None,
                   repeat: bool = False,
                   restart: bool = False):
        assert not (batch_per_epoch is None and repeat), 'this iter won\'t stop'
        samples = []
        bs = 0
        if restart:  # restart iteration
            self.renew_data_iter(split)
        iterator = self.get_data_iter(split)
        while True:
            try:
                s = next(iterator)
                samples.append(s)
                if len(samples) >= batch_size:
                    yield samples
                    bs += 1
                    if batch_per_epoch is not None and bs >= batch_per_epoch:
                        break
                    samples = []
            except StopIteration:
                if repeat:
                    self.renew_data_iter(split)
                    iterator = self.get_data_iter(split)
                else:
                    if len(samples) > 0:
                        yield samples
                        samples = []
                    break
