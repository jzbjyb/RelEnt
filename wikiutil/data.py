from typing import Tuple, List, Any, Dict
from collections import defaultdict, namedtuple
from random import shuffle
from tqdm import tqdm
from functools import lru_cache
import numpy as np
import multiprocessing
from multiprocessing import Manager
import torch
from torch.utils.data import Dataset
from .property import read_subgraph_file, read_multi_pointiwse_file


class DataPrepError(Exception):
    pass


class PropertySubgraph():
    def __init__(self,
                 property_id: str,
                 occurrences: Tuple[Tuple[str, str]],
                 subgraph_dict: Dict[str, List[Tuple[str, str, str]]],
                 id2ind: Dict[str, int] = None,
                 padding_ind: int = 0,  # TODO: add padding index
                 edge_type: str = 'one'):
        self.pid = property_id
        self.occurrences = occurrences
        self.subgraph_dict = subgraph_dict
        self.id2ind = id2ind
        self.padding_ind = padding_ind
        assert edge_type in {'one'}
        self.edge_type = edge_type
        self.id2ind, self.adjs, self.emb_ind, self.prop_ind = self._to_gnn_input()


    @property
    def size(self):
        return len(self.id2ind)


    def _to_gnn_input(self):
        ''' convert to gnn input format (adjacency list, input emb index, and prop index) '''

        if self.edge_type == 'one':
            # build adj list
            id2ind = defaultdict(lambda: len(id2ind))  # entity/proerty id to index mapping
            _ = id2ind[self.pid]  # make sure that the central property is alway indexed as 0
            adjs = []
            for i, (hid, tid) in enumerate(self.occurrences):
                # for each occurrence, create a pseudo property and link it to the real one
                # TODO: (1) use another type of link to represent
                #  the connection between pseudo property and real property?
                # TODO: (2) treat property as a node is problematic because the same property might
                #  appear multiple times in the subgraph
                ppid = self.pid + '-' + str(i)
                adjs.append((id2ind[ppid], id2ind[self.pid]))  # link pseudo property to real property
                adjs.append((id2ind[hid], id2ind[ppid]))
                adjs.append((id2ind[ppid], id2ind[tid]))
                try:
                    for two_side in [self.subgraph_dict[hid], self.subgraph_dict[tid]]:
                        for e1, p, e2 in two_side:
                            e1 = id2ind[e1]
                            p = id2ind[p]
                            e2 = id2ind[e2]
                            adjs.append((e1, p))
                            adjs.append((p, e2))
                except KeyError:
                    raise DataPrepError

            # build emb index
            ind2id = dict((v, k) for k, v in id2ind.items())
            if self.id2ind:
                try:
                    emb_ind = np.array([self.id2ind[ind2id[i].split('-')[0]] for i in range(len(id2ind))])
                except KeyError:
                    raise DataPrepError
            else:
                # when id2ind does not exist, use padding_ind
                emb_ind = np.array([self.padding_ind] * len(id2ind))

            return id2ind, adjs, emb_ind, [0]


    @staticmethod
    def pack_graphs(graphs: List):
        ''' convert to gnn input format (adjacency list, input emb, and prop index) '''
        if len(graphs) == 0:
            raise Exception
        edge_type = graphs[0].edge_type
        if edge_type == 'one':
            new_adjs, new_emb_ind, new_prop_ind = [], [], []
            acc = 0
            for g in graphs:
                for adj in g.adjs:
                    new_adjs.append((adj[0] + acc, adj[1] + acc))
                new_emb_ind.append(g.emb_ind)
                new_prop_ind.append(acc)
                acc += g.size
            new_emb = np.concatenate(new_emb_ind, axis=0)
            return new_adjs, new_emb, new_prop_ind


class AdjacencyList:
    ''' represent the topology of a graph '''
    def __init__(self, node_num: int, adj_list: List):
        self.node_num = node_num
        self.data = torch.tensor(adj_list, dtype=torch.long)
        self.edge_num = len(adj_list)


    @property
    def device(self):
        return self.data.device


    def to(self, device):
        self.data = self.data.to(device)
        return self


    def __getitem__(self, item):
        return self.data[item]


class PointwiseDataset(Dataset):
    def __init__(self,
                 filepath: str,
                 subgraph_dict: Dict[str, List],
                 id2ind: Dict[str, int] = None,
                 padding_ind: int = 0,
                 edge_type: str = 'one',
                 filter_prop: set = None,
                 keep_one_per_prop: bool = False,
                 neg_ratio: int = None,
                 manager: Manager = None):
        print('load data from {} ...'.format(filepath))
        self.inst_list = read_multi_pointiwse_file(
            filepath, filter_prop=filter_prop, keep_one_per_prop=keep_one_per_prop)
        if neg_ratio:
            self.inst_list = self.pos_neg_filter(self.inst_list, neg_ratio=neg_ratio)
        self.subgraph_dict = subgraph_dict
        assert edge_type in {'one'}
        # 'one' means only use a single type of edge to link properties and entities
        self.edge_type = edge_type
        self.id2ind = id2ind
        self.padding_ind = padding_ind

        # to support multiprocess, use manager to share these objects between processes and avoid copy-on-write
        # manager is quite slow because of the communication between processes
        if manager:
            assert type(id2ind) is type(subgraph_dict) is multiprocessing.managers.DictProxy
            self.inst_list = manager.list(self.inst_list)
        # TODO: use numpy array or shared array to avoid copy-on-write


    def pos_neg_filter(self, samples: List[Tuple[Any, Any, int]], neg_ratio: int = 3):
        # used to filter negative samples for pointwise methods
        pos = [s for s in samples if s[2] > 0]
        neg = [s for s in samples if s[2] == 0]
        shuffle(neg)
        return pos + neg[:len(pos) * neg_ratio]


    def __len__(self):
        return len(self.inst_list)


    def __getitem__(self, idx):
        p1o, p2o, label = self.inst_list[idx]
        return self.create_subgraph(p1o), self.create_subgraph(p2o), label


    #@lru_cache(maxsize=10000)  # result in out-of-memory error
    def create_subgraph(self, property_occ: Tuple[str, Tuple[Tuple[str, str]]]) -> PropertySubgraph:
        pid, occs = property_occ
        sg = PropertySubgraph(
            pid, occs, self.subgraph_dict, self.id2ind, self.padding_ind, self.edge_type)
        return sg


    def collate_fn(self, insts: List):
        packed_sg1, packed_sg2, labels = zip(*insts)
        pid1s = [sg.pid for sg in packed_sg1]
        pid2s = [sg.pid for sg in packed_sg2]
        adjs12, emb_ind12, prop_ind12 = [], [], []
        for packed_sg in [packed_sg1, packed_sg2]:
            adjs, emb_ind, prop_ind = PropertySubgraph.pack_graphs(packed_sg)
            adjs = AdjacencyList(node_num=len(adjs), adj_list=adjs)
            emb_ind = torch.LongTensor(emb_ind)
            prop_ind = torch.LongTensor(prop_ind)
            adjs12.append(adjs)
            emb_ind12.append(emb_ind)
            prop_ind12.append(prop_ind)
        return {'adj': adjs12, 'emb_ind': emb_ind12, 'prop_ind': prop_ind12,
                'meta': {'pid1': pid1s, 'pid2': pid2s}}, \
               torch.LongTensor(labels)


class PointwiseDataLoader():
    def __init__(self,
                 train_file: str,
                 dev_file: str,
                 test_file: str,
                 subgraph_file: str,
                 id2ind: Dict[str, int] = None,
                 padding_ind: int = 0,
                 edge_type: str = 'one',
                 filter_prop: set = None,
                 keep_one_per_prop: bool = False,
                 num_worker: int = 1,
                 neg_ratio: int = 0):
        print('load data ...')
        self.train_list = read_multi_pointiwse_file(
            train_file, filter_prop=filter_prop, keep_one_per_prop=keep_one_per_prop)
        if neg_ratio:
            self.train_list = self.pos_neg_filter(self.train_list, neg_ratio=neg_ratio)
        self.dev_list = read_multi_pointiwse_file(
            dev_file, filter_prop=filter_prop, keep_one_per_prop=keep_one_per_prop)
        self.test_list = read_multi_pointiwse_file(
            test_file, filter_prop=filter_prop, keep_one_per_prop=keep_one_per_prop)
        self.subgraph_dict = read_subgraph_file(subgraph_file)
        assert edge_type in {'one'}
        # 'one' means only use a single type of edge to link properties and entities
        self.edge_type = edge_type
        self.id2ind = id2ind
        self.padding_ind = padding_ind
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


    def create_subgraph(self, property_occ: Tuple[str, List[Tuple[str, str]]]) -> PropertySubgraph:
        pid, occs = property_occ
        property_occ_hash = pid + ':' + ','.join(map(lambda occ: '-'.join(occ), occs))
        if property_occ_hash not in self._subgraph_cache:
            sg = PropertySubgraph(
                pid, occs, self.subgraph_dict, self.id2ind, self.padding_ind, self.edge_type)
            # TODO: add multiprocess with lock
            #with self._cache_lock.get_lock():
            if property_occ_hash not in self._subgraph_cache:
                self._subgraph_cache[property_occ_hash] = sg
                self.all_ids.update(set(sg.id2ind.keys()))
        return self._subgraph_cache[property_occ_hash]


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
