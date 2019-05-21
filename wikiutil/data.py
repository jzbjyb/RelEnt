from typing import Tuple, List, Any, Dict
from collections import defaultdict, namedtuple
from random import shuffle
from tqdm import tqdm
import numpy as np
import multiprocessing
import torch
from torch.utils.data import Dataset
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
        self.id2ind, self.adjs, self.emb, self.prop_ind = self._to_gnn_input(emb_dict, emb_size)


    @property
    def size(self):
        return len(self.id2ind)


    def _to_gnn_input(self,
                     emb_dict: Dict[str, List[float]] = None,
                     emb_size: int = None):
        ''' convert to gnn input format (adjacency list, input emb, and prop index) '''

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

            return id2ind, adjs, emb, [0]


    @staticmethod
    def pack_graphs(graphs: List):
        ''' convert to gnn input format (adjacency list, input emb, and prop index) '''
        if len(graphs) == 0:
            raise Exception
        edge_type = graphs[0].edge_type
        if edge_type == 'one':
            new_adjs, new_emb, new_prop_ind = [], [], []
            acc = 0
            for g in graphs:
                for adj in g.adjs:
                    new_adjs.append((adj[0] + acc, adj[1] + acc))
                new_emb.append(g.emb)
                new_prop_ind.append(acc)
                acc += g.size
            new_emb = np.concatenate(new_emb, axis=0)
            return new_adjs, new_emb, new_prop_ind


class AdjacencyList:
    """represent the topology of a graph"""
    def __init__(self, node_num: int, adj_list: List, device: torch.device = None):
        self.node_num = node_num
        self.data = torch.tensor(adj_list, dtype=torch.long, device=device)
        self.edge_num = len(adj_list)


    @property
    def device(self):
        return self.data.device


    def to(self, device):
        self.data.to(device)


    def __getitem__(self, item):
        return self.data[item]


class PointwiseDataset(Dataset):
    def __init__(self,
                 filepath: str,
                 subgraph_file: str,
                 emb_size: int = None,
                 emb_dict: str = None,
                 edge_type: str = 'one',
                 filter_prop: set = None,
                 keep_one_per_prop: bool = False,
                 neg_ratio: int = None):
        print('load data from {} ...'.format(filepath))
        self.inst_list = read_pointiwse_file(
            filepath, filter_prop=filter_prop, keep_one_per_prop=keep_one_per_prop)
        if neg_ratio:
            self.inst_list = self.pos_neg_filter(self.inst_list, neg_ratio=neg_ratio)
        self.subgraph_dict = read_subgraph_file(subgraph_file)
        assert edge_type in {'one'}
        # 'one' means only use a single type of edge to link properties and entities
        self.edge_type = edge_type
        self.emb_size = emb_size
        self.emb_dict = emb_dict


    def pos_neg_filter(self, samples: List[Tuple[Any, Any, int]], neg_ratio: int = 3):
        # used to filter negative samples for pointwise methods
        pos = [s for s in samples if s[2] > 0]
        neg = [s for s in samples if s[2] == 0]
        shuffle(neg)
        return pos + neg[:len(pos) * neg_ratio]


    def __len__(self):
        return len(self.inst_list)


    def __getitem__(self, idx):
        return self.inst_list[idx]


    def create_subgraph(self, property_occ: Tuple[str, str, str]) -> PropertySubgraph:
        hid, pid, tid = property_occ
        sg = PropertySubgraph(
            pid, hid, tid, self.subgraph_dict, self.emb_dict, self.emb_size, self.edge_type)
        return sg


    def collate_fn(self, insts: List):
        result = []
        for inst in insts:
            p1o, p2o, label = inst
            sample = (self.create_subgraph(p1o), self.create_subgraph(p2o), label)
            result.append(sample)
        packed_sg1, packed_sg2, labels = zip(*result)
        adjs12, e12, prop_ind12 = [], [], []
        for packed_sg in [packed_sg1, packed_sg2]:
            adjs, e, prop_ind = PropertySubgraph.pack_graphs(packed_sg)
            adjs = AdjacencyList(node_num=len(adjs), adj_list=adjs)
            e = torch.tensor(e, dtype=torch.float)
            prop_ind = torch.tensor(prop_ind)
            adjs12.append(adjs)
            e12.append(e)
            prop_ind12.append(prop_ind)
        return {'adj': adjs12, 'emb': e12, 'prop_ind': prop_ind12}, \
               torch.LongTensor(labels)


class PointwiseDataLoader():
    def __init__(self,
                 train_file: str,
                 dev_file: str,
                 test_file: str,
                 subgraph_file: str,
                 emb_size: int = None,
                 emb_file: str = None,
                 edge_type: str = 'one',
                 filter_prop: set = None,
                 keep_one_per_prop: bool = False,
                 num_worker: int = 1):
        print('load data ...')
        self.train_list = read_pointiwse_file(
            train_file, filter_prop=filter_prop, keep_one_per_prop=keep_one_per_prop)
        self.train_list = self.pos_neg_filter(self.train_list, neg_ratio=5)  # TODO: add neg_ratio param
        self.dev_list = read_pointiwse_file(
            dev_file, filter_prop=filter_prop, keep_one_per_prop=keep_one_per_prop)
        self.test_list = read_pointiwse_file(
            test_file, filter_prop=filter_prop, keep_one_per_prop=keep_one_per_prop)
        self.subgraph_dict = read_subgraph_file(subgraph_file)
        assert edge_type in {'one'}
        # 'one' means only use a single type of edge to link properties and entities
        self.edge_type = edge_type
        self.emb_size = emb_size
        self.emb_dict = load_embedding(emb_file) if emb_file else None
        print('prep data ...')
        self.all_ids = set()  # all the entity ids and property ids used
        self._subgraph_cache: Dict[Tuple[str, str, str], PropertySubgraph] = {}  # cache subgraphs
        self._cache_lock = multiprocessing.Value('i', 0)
        self.num_worker = num_worker
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


    def create_pointwise_samples_mutiple_process(self, data: List):
        # start each worker with a shared return_dict
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        workers = []
        len_per_worker = int(np.ceil(len(data) / self.num_worker))
        for i in range(self.num_worker):
            data_for_worker = data[i * len_per_worker:(i + 1) * len_per_worker]
            workder = multiprocessing.Process(target=self.create_pointwise_samples,
                                              args=(data_for_worker, i, return_dict))
            workers.append(workder)
            workder.start()

        # wait for end of each worker
        for worker in workers:
            worker.join()

        # merge results
        result = []
        for i in range(self.num_worker):
            result.extend(return_dict[i])


    def create_pointwise_samples(self, data: List, worker_id: int = None, return_dict: Dict = None):
        result = []
        for sample in tqdm(data):
            p1o, p2o, label = sample
            sample = (self.create_subgraph(p1o), self.create_subgraph(p2o), label)
            result.append(sample)
        if worker_id is not None:
            return_dict[worker_id] = result
        return result


    def create_subgraph(self, property_occ: Tuple[str, str, str]) -> PropertySubgraph:
        hid, pid, tid = property_occ

        if property_occ not in self._subgraph_cache:
            sg = PropertySubgraph(
                pid, hid, tid, self.subgraph_dict, self.emb_dict, self.emb_size, self.edge_type)
            # TODO: add multiprocess with lock
            #with self._cache_lock.get_lock():
            if property_occ not in self._subgraph_cache:
                self._subgraph_cache[property_occ] = sg
                self.all_ids.update(set(sg.id2ind.keys()))

        return self._subgraph_cache[property_occ]


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
