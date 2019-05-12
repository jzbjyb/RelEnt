from typing import Tuple, List, Any, Dict
from collections import defaultdict
from random import shuffle
from tqdm import tqdm
import numpy as np
from .property import read_pointiwse_file, read_subgraph_file


class DataPrepError(Exception):
    pass


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
        self.train_list = self.pos_neg_filter(self.train_list, neg_ratio=5)  # TODO: neg_ratio param
        self.dev_list = read_pointiwse_file(dev_file)
        self.test_list = read_pointiwse_file(test_file)
        self.subgraph_dict = read_subgraph_file(subgraph_file)
        assert edge_type in {'one'}
        # 'one' means only use a single type of edge to link properties and entities
        self.edge_type = edge_type
        self.emb_size = emb_size
        if emb_file:
            self.emb_dict = load_embedding(emb_file)
        print('prep data ...')
        self.all_ids = set()  # all the entity ids and property ids used
        self.train_graph = self.create_pointwise_samples(self.train_list)
        print('train {} -> {}'.format(len(self.train_list), len(self.train_graph)))
        self.dev_graph = self.create_pointwise_samples(self.dev_list)
        print('dev {} -> {}'.format(len(self.dev_list), len(self.dev_graph)))
        self.test_graph = self.create_pointwise_samples(self.test_list)
        print('test {} -> {}'.format(len(self.test_list), len(self.test_graph)))


    def create_pointwise_samples(self, data: List):
        result = []
        for sample in tqdm(data):
            try:
                sample = self.create_one_pointwise_sample(*sample)
                result.append(sample)
            except DataPrepError:
                continue
        return result


    def create_one_pointwise_sample(self,
                          property_occ1: Tuple[str, str, str],
                          property_occ2: Tuple[str, str, str],
                          label: int):
        return self.create_subgraph(property_occ1), self.create_subgraph(property_occ2), label


    def pos_neg_filter(self, samples: List[Tuple[Any, Any, int]], neg_ratio: int = 3):
        # used to filter negative samples for pointwise methods
        pos = [s for s in samples if s[2] > 0]
        neg = [s for s in samples if s[2] == 0]
        shuffle(neg)
        return pos + neg[:len(pos) * neg_ratio]


    def create_subgraph(self, property_occ: Tuple[str, str, str]) \
            -> Tuple[np.ndarray, List]:  # return init emb and adj list
        id2ind = defaultdict(lambda: len(id2ind))  # entity/proerty id to index mapping
        pid, hid, tid = property_occ
        _ = id2ind[pid]  # make sure that the central property is alway indexed as 0

        # subgraph not exist
        if hid not in self.subgraph_dict or tid not in self.subgraph_dict:
            raise DataPrepError

        # build emb and adj list
        if self.edge_type == 'one':
            adj_list = []
            adj_list.extend([(id2ind[hid], id2ind[pid]), (id2ind[pid], id2ind[tid])])
            for two_side in [hid, tid]:
                for e1, p, e2 in self.subgraph_dict[two_side]:
                    e1 = id2ind[e1]
                    p = id2ind[p]
                    e2 = id2ind[e2]
                    adj_list.append((e1, p))
                    adj_list.append((p, e2))
            ind2id = dict((v, k) for k, v in id2ind.items())
            if hasattr(self, 'emb_dict'):
                try:
                    emb = np.array([self.emb_dict[ind2id[i]] for i in range(len(id2ind))])
                except KeyError:
                    raise DataPrepError
            else:
                emb = np.ones((len(id2ind), self.emb_size))

        # collect all ids
        self.all_ids |= id2ind.keys()
        return emb, adj_list


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
                    break
