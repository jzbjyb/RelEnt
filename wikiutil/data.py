from typing import Tuple, List, Any, Dict
from random import shuffle
from tqdm import tqdm
from functools import lru_cache
import numpy as np
import multiprocessing
from multiprocessing import Manager
import torch
from torch.utils.data import Dataset
from .property import read_multi_pointiwse_file, read_nway_file
from .util import DefaultOrderedDict


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
            id2ind = DefaultOrderedDict(lambda: len(id2ind))  # entity/proerty id to index mapping
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
            if self.id2ind:
                try:
                    emb_ind = np.array([self.id2ind[k.split('-')[0]] for k, v in id2ind.items()])
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


class PropertyDataset(Dataset):
    def __init__(self,
                 subgraph_dict: Dict[str, List],
                 id2ind: Dict[str, int] = None,
                 padding_ind: int = 0,
                 edge_type: str = 'one',
                 use_cache: bool = False):
        self.subgraph_dict = subgraph_dict
        assert edge_type in {'one'}
        # 'one' means only use a single type of edge to link properties and entities
        self.edge_type = edge_type
        self.id2ind = id2ind
        self.padding_ind = padding_ind
        self.use_cache = use_cache
        self._cache = {}


    def pos_neg_filter(self, samples: List[Tuple], neg_ratio: int = 3):
        pos = [s for s in samples if s[-1] > 0]
        neg = [s for s in samples if s[-1] == 0]
        shuffle(neg)
        return pos + neg[:len(pos) * neg_ratio]


    def getids(self, getitem_result) -> set:
        ''' get the ids from the return of __getitem__ '''
        raise NotImplementedError


    def collect_ids(self) -> set:
        print('collecting ids ...')
        ids = set()
        for i in tqdm(range(self.__len__())):
            ids.update(self.getids(self.__getitem__(i)))
        return ids


    #@lru_cache(maxsize=10000)  # result in out-of-memory error
    def create_subgraph(self, property_occ: Tuple[str, Tuple[Tuple[str, str]]]) -> PropertySubgraph:
        if self.use_cache and property_occ in self._cache:
            return self._cache[property_occ]
        pid, occs = property_occ
        sg = PropertySubgraph(
            pid, occs, self.subgraph_dict, self.id2ind, self.padding_ind, self.edge_type)
        if self.use_cache:
            self._cache[property_occ] = sg
        return sg


class PointwiseDataset(PropertyDataset):
    def __init__(self,
                 filepath: str,
                 subgraph_dict: Dict[str, List],
                 id2ind: Dict[str, int] = None,
                 padding_ind: int = 0,
                 edge_type: str = 'one',
                 use_cache: bool = False,
                 filter_prop: set = None,
                 keep_one_per_prop: bool = False,
                 neg_ratio: int = None,
                 manager: Manager = None):
        super(PointwiseDataset, self).__init__(subgraph_dict, id2ind, padding_ind, edge_type, use_cache)
        print('load data from {} ...'.format(filepath))
        self.inst_list = read_multi_pointiwse_file(
            filepath, filter_prop=filter_prop, keep_one_per_prop=keep_one_per_prop)
        if neg_ratio:
            self.inst_list = self.pos_neg_filter(self.inst_list, neg_ratio=neg_ratio)

        # to support multiprocess, use manager to share these objects between processes and avoid copy-on-write
        # manager is quite slow because of the communication between processes
        if manager:
            assert type(id2ind) is type(subgraph_dict) is multiprocessing.managers.DictProxy
            self.inst_list = manager.list(self.inst_list)
        # TODO: use numpy array or shared array to avoid copy-on-write


    def __len__(self):
        return len(self.inst_list)


    def __getitem__(self, idx):
        p1o, p2o, label = self.inst_list[idx]
        return self.create_subgraph(p1o), self.create_subgraph(p2o), label


    def getids(self, getitem_result):
        g1, g2, _ = getitem_result
        return set(g1.id2ind.keys()) | set(g2.id2ind.keys())


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


class NwayDataset(PropertyDataset):
    def __init__(self,
                 filepath: str,
                 subgraph_dict: Dict[str, List],
                 id2ind: Dict[str, int] = None,
                 padding_ind: int = 0,
                 edge_type: str = 'one',
                 use_cache: bool = False,
                 filter_prop: set = None,
                 keep_one_per_prop: bool = False):
        super(NwayDataset, self).__init__(subgraph_dict, id2ind, padding_ind, edge_type, use_cache)
        print('load data from {} ...'.format(filepath))
        self.inst_list = read_nway_file(
            filepath, filter_prop=filter_prop, keep_one_per_prop=keep_one_per_prop)

        # TODO: use numpy array or shared array to avoid copy-on-write


    def __len__(self):
        return len(self.inst_list)


    def __getitem__(self, idx):
        poccs, label = self.inst_list[idx]
        return self.create_subgraph(poccs), label


    def getids(self, getitem_result):
        g, _ = getitem_result
        return set(g.id2ind.keys())


    def collate_fn(self, insts: List):
        packed_sg, labels = zip(*insts)
        pids = [sg.pid for sg in packed_sg]
        adjs, emb_ind, prop_ind = PropertySubgraph.pack_graphs(packed_sg)
        adjs = AdjacencyList(node_num=len(adjs), adj_list=adjs)
        emb_ind = torch.LongTensor(emb_ind)
        prop_ind = torch.LongTensor(prop_ind)
        return {'adj': [adjs], 'emb_ind': [emb_ind], 'prop_ind': [prop_ind],
                'meta': {'pid': pids}}, \
               torch.LongTensor(labels)
