from typing import Tuple, List, Any, Dict, Iterable
from random import shuffle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import json
import numpy as np
import multiprocessing
from multiprocessing import Manager
import torch
from torch.utils.data import Dataset
from .property import read_multi_pointiwse_file, read_nway_file
from .util import DefaultOrderedDict, load_tsv_as_dict
from .constant import AGG_NODE, AGG_PROP, PADDING


TOP_ENTITY_IDS = set(load_tsv_as_dict('data/entity_count_5000.tsv').keys())
TOP_PROP_IDS = set(load_tsv_as_dict('data/property_count_200.tsv').keys())


def load_lines(filepath) -> List[str]:
    with open(filepath, 'r') as fin:
        result = [l.strip() for l in fin]
    return result


class DataPrepError(Exception):
    pass


class PropertySubgraph():
    def __init__(self,
                 property_id: str = None,
                 occurrences: Tuple[Tuple[str, str]] = None,
                 subgraph_dict: Dict[str, List[Tuple[str, str, str]]] = None,
                 emb_id2ind: Dict[str, int] = None,
                 padding: str = PADDING,  # TODO: add padding
                 edge_type: str = 'one',
                 use_pseudo_property: bool = False,
                 agg_property: str = AGG_PROP,
                 merge: bool = False,
                 property_id2: str = None,
                 occurrences2: Tuple[Tuple[str, str]] = None,
                 use_top: bool = False):
        if not property_id:
            return
        self.pid = property_id
        self.pid2 = property_id2
        self.occurrences = occurrences
        self.occurrences2 = occurrences2
        self.subgraph_dict = subgraph_dict
        self.emb_id2ind = emb_id2ind
        self.padding = padding
        assert edge_type in {'one', 'property', 'only_property', 'bow'}
        # 'one': properties are treated as nodes and only use a single type of edges
        # 'property': properties are treated as edges
        # 'only_keep_properties': only keep properties in the subgraph
        # 'bow': treat properties and entites as a bag of words
        self.edge_type = edge_type
        self.use_pseudo_property = use_pseudo_property
        self.agg_property = agg_property  # a special property used to aggregate information
        self.merge = merge  # merge subgraphs of two properties
        self.use_top = use_top  # if True, only use top properties and entities  # TODO: add use_top to all edge_type
        self.id2ind, self.ind2adjs, self.emb_ind, self.prop_ind = self._to_gnn_input()
        self.ids = set([k.split('-')[0] for k in self.id2ind])  # all the ids of entities and properties involved


    @property
    def size(self):
        return len(self.emb_ind)


    @classmethod
    def from_string(cls, format_str, edge_type):
        '''
        recover necessary properties from a formatted string.
        necessary properties: pid, ind2adjs, emb_ind, prop_ind
        '''
        empty_sg = cls()
        pid, ind2adjs, emb_ind, prop_ind = format_str.split('\t')
        if pid.find('|') >= 0:
            pid = pid.split('|')
            empty_sg.pid = pid[0]
            empty_sg.pid2 = pid[1]
        else:
            empty_sg.pid = pid
        empty_sg.ind2adjs = json.loads(ind2adjs)
        empty_sg.emb_ind = np.array([i for i in map(int, emb_ind.split(','))])
        empty_sg.prop_ind = int(prop_ind)
        empty_sg.edge_type = edge_type
        return empty_sg


    def to_string(self):
        '''
        dump necessary properties to a formatted string.
        '''
        pid = self.pid
        if self.pid2 is not None:
            pid += '|' + self.pid2
        return '{pid}\t{ind2adjs}\t{emb_ind}\t{prop_ind}'.format(
            pid=pid,
            ind2adjs=json.dumps(self.ind2adjs),
            emb_ind=','.join(str(i) for i in self.emb_ind),
            prop_ind=self.prop_ind)


    def get_pseudo_property_id(self, pid: str, head_id: str, tail_id: str, pid2count: Dict[str, int]):
        '''
        The minimal number of pseudo properties for each property is the minimum of the following four:
        1. the number of unique head entities.
        2. the number of unique tail entities.
        3. the number of unique head entities with different tail node set (<= 1).
        4. the number of unique tail entities with different head node set (<= 2).
        '''
        uniq_pid = pid + '-' + head_id
        #if uniq_pid not in pid2count:
        #    pid2count[uniq_pid] = 0
        #pid2count[uniq_pid] += 1
        #uniq_pid = '{}-{}'.format(pid, pid2count[uniq_pid])
        return uniq_pid


    def build_identical_link(self,
                             id: str,
                             old_set: set,
                             id2ind: Dict[str, int],
                             adjs: List[Tuple[int, int]]):
        if id in old_set:
            nid = id + '-2'
            adjs.append((id2ind[nid], id2ind[id]))
            return nid
        return id


    def is_pass_check(self, id):
        if not self.use_top:
            return True
        if id.startswith('P'):
            if id not in TOP_PROP_IDS:
                return False
        elif id.startswith('Q'):
            if id not in TOP_ENTITY_IDS:
                return False
        else:
            raise Exception
        return True


    def normalize(self, id):
        ''' use padding to replace infrequent entities or properties '''
        return id if self.is_pass_check(id) else self.padding


    def _to_gnn_input(self):
        ''' convert to gnn input format (adjacency list, input emb index, and prop index) '''

        if self.edge_type == 'one':
            # build adj list
            id2ind = DefaultOrderedDict(lambda: len(id2ind))  # entity/proerty id to index mapping
            pid2count = {}  # used to generate a pseudo property id for each occurrence of a property
            _ = id2ind[self.pid]  # make sure that the central property is alway indexed as 0
            adjs = []
            adjs_dict = {'one': adjs}
            for i, (hid, tid) in enumerate(self.occurrences):
                # for each occurrence, create a pseudo property and link it to the real one
                # TODO: (1) use another type of link to represent
                #  the connection between pseudo property and real property?
                # TODO: (2) treat property as a node is problematic because the same property might
                #  appear multiple times in the subgraph
                # TODO: (3) it might be better to treat the links from property to entity and
                #  the links from entity to property differently
                if self.use_pseudo_property:  # use a pseudo prop and link it to the real one
                    ppid = self.get_pseudo_property_id(self.pid, hid, tid, pid2count)
                    adjs.append((id2ind[ppid], id2ind[self.pid]))
                else:
                    ppid = self.pid
                adjs.append((id2ind[self.normalize(hid)], id2ind[ppid]))
                adjs.append((id2ind[ppid], id2ind[self.normalize(tid)]))
                try:
                    for two_side in [self.subgraph_dict[hid], self.subgraph_dict[tid]]:
                        # TODO: pid could be pid2, which might be cheating
                        for e1, pid, e2 in two_side:
                            e1 = self.normalize(e1)
                            pid = self.normalize(pid)
                            e2 = self.normalize(e2)
                            if self.use_pseudo_property:  # use a pseudo prop and link it to the real one
                                ppid = self.get_pseudo_property_id(pid, e1, e2, pid2count)
                                adjs.append((id2ind[ppid], id2ind[pid]))
                            else:
                                ppid = pid
                            adjs.append((id2ind[e1], id2ind[ppid]))
                            adjs.append((id2ind[ppid], id2ind[e2]))
                except KeyError:
                    raise DataPrepError

            # merge
            if self.merge:
                adjs_dict['pair'] = [(id2ind[self.pid2], id2ind[self.pid])]
                for i, (hid, tid) in enumerate(self.occurrences2):
                    if self.use_pseudo_property:
                        raise NotImplementedError
                    adjs.append((id2ind[self.normalize(hid)], id2ind[self.pid2]))
                    adjs.append((id2ind[self.pid2], id2ind[self.normalize(tid)]))
                    try:
                        for two_side in [self.subgraph_dict[hid], self.subgraph_dict[tid]]:
                            for e1, pid, e2 in two_side:
                                e1 = self.normalize(e1)
                                pid = self.normalize(pid)
                                e2 = self.normalize(e2)
                                adjs.append((id2ind[e1], id2ind[pid]))
                                adjs.append((id2ind[pid], id2ind[e2]))
                    except KeyError:
                        raise DataPrepError

            # remove duplicate item in adjs
            adjs_dict = dict((k, list(set(v))) for k, v in adjs_dict.items())

            # build emb index
            if self.emb_id2ind:
                try:
                    emb_ind = np.array([self.emb_id2ind[k.split('-')[0]] for k, v in id2ind.items()])
                except KeyError:
                    raise DataPrepError
            else:
                emb_ind = np.array([0] * len(id2ind))

            return id2ind, adjs_dict, emb_ind, 0

        elif self.edge_type == 'only_property':  # TODO: add use_top
            # build adj list
            id2ind = DefaultOrderedDict(lambda: len(id2ind))  # proerty id to index mapping
            _ = id2ind[self.pid]  # make sure that the central property is alway indexed as 0
            adjs_dict: Dict[str, List] = {'head': [], 'tail': []}
            for i, (hid, tid) in enumerate(self.occurrences):
                try:
                    for e1, pid, e2 in self.subgraph_dict[hid]:
                        # TODO: better way to avoid treating
                        if pid != self.pid and (not self.pid2 or pid != self.pid2):
                            adjs_dict['head'].append((id2ind[pid], id2ind[self.pid]))
                    for e1, pid, e2 in self.subgraph_dict[tid]:
                        if pid != self.pid and (not self.pid2 or pid != self.pid2):
                            adjs_dict['tail'].append((id2ind[pid], id2ind[self.pid]))
                except KeyError:
                    raise DataPrepError

            if self.merge:
                if self.pid2 in id2ind:
                    raise Exception('pid2 is included, might be cheating')
                adjs_dict['pair'] = [(id2ind[self.pid2], id2ind[self.pid])]
                for i, (hid, tid) in enumerate(self.occurrences2):
                    try:
                        for e1, pid, e2 in self.subgraph_dict[hid]:
                            if pid != self.pid2:
                                adjs_dict['head'].append((id2ind[pid], id2ind[self.pid2]))
                        for e1, pid, e2 in self.subgraph_dict[tid]:
                            if pid != self.pid2:
                                adjs_dict['tail'].append((id2ind[pid], id2ind[self.pid2]))
                    except KeyError:
                        raise DataPrepError

            adjs_dict = dict((k, list(set(v))) for k, v in adjs_dict.items())

            # build emb index
            if self.emb_id2ind:
                try:
                    emb_ind = np.array([self.emb_id2ind[k.split('-')[0]] for k, v in id2ind.items()])
                except KeyError:
                    raise DataPrepError
            else:
                emb_ind = np.array([0] * len(id2ind))

            return id2ind, adjs_dict, emb_ind, 0

        elif self.edge_type == 'bow':
            # build adj list
            id2ind = DefaultOrderedDict(lambda: len(id2ind))  # entity/proerty id to index mapping
            pi = id2ind[self.pid]  # make sure that the central property is alway indexed as 0
            adjs_dict: Dict[str, List] = {'head': [], 'tail': [], 'head_other': [], 'tail_other': []}
            for i, (hid, tid) in enumerate(self.occurrences):
                adjs_dict['head'].append((id2ind[self.normalize(hid)], pi))
                adjs_dict['tail'].append((id2ind[self.normalize(tid)], pi))
                try:
                    # TODO: avoid treating
                    for e1, pid, e2 in self.subgraph_dict[hid]:
                        if pid == self.pid:
                            continue
                        adjs_dict['head_other'].append((id2ind[self.normalize(pid)], pi))
                        adjs_dict['head_other'].append((id2ind[self.normalize(e2)], pi))
                    for e1, pid, e2 in self.subgraph_dict[tid]:
                        if pid == self.pid:
                            continue
                        adjs_dict['tail_other'].append((id2ind[self.normalize(pid)], pi))
                        adjs_dict['tail_other'].append((id2ind[self.normalize(e2)], pi))
                except KeyError:
                    raise DataPrepError

            if self.merge:
                pi2 = id2ind[self.pid2]
                adjs_dict['pair'] = [(pi2, pi)]
                for i, (hid, tid) in enumerate(self.occurrences2):
                    adjs_dict['head'].append((id2ind[self.normalize(hid)], pi2))
                    adjs_dict['tail'].append((id2ind[self.normalize(tid)], pi2))
                    try:
                        for e1, pid, e2 in self.subgraph_dict[hid]:
                            if pid == self.pid2:
                                continue
                            adjs_dict['head'].append((id2ind[self.normalize(e1)], pi2))
                            adjs_dict['head'].append((id2ind[self.normalize(pid)], pi2))
                            adjs_dict['head'].append((id2ind[self.normalize(e2)], pi2))
                        for e1, pid, e2 in self.subgraph_dict[tid]:
                            if pid == self.pid2:
                                continue
                            adjs_dict['tail'].append((id2ind[self.normalize(e1)], pi2))
                            adjs_dict['tail'].append((id2ind[self.normalize(pid)], pi2))
                            adjs_dict['tail'].append((id2ind[self.normalize(e2)], pi2))
                    except KeyError:
                        raise DataPrepError

            adjs_dict = dict((k, list(set(v))) for k, v in adjs_dict.items())

            # build emb index
            if self.emb_id2ind:
                try:
                    emb_ind = np.array([self.emb_id2ind[k.split('-')[0]] for k, v in id2ind.items()])
                except KeyError:
                    raise DataPrepError
            else:
                emb_ind = np.array([0] * len(id2ind))

            return id2ind, adjs_dict, emb_ind, 0

        elif self.edge_type == 'property':
            # build adj list
            id2ind = DefaultOrderedDict(lambda: len(id2ind))  # entity id to index mapping

            pid2adjs = DefaultOrderedDict(lambda: [])
            _ = id2ind[AGG_NODE]  # make sure the agg node is the first one
            for i, (hid, tid) in enumerate(self.occurrences):
                # TODO: use which node to gather propagated information?
                agg_adjs = pid2adjs[self.agg_property]
                agg_adjs.append((id2ind[hid], id2ind[AGG_NODE]))
                agg_adjs.append((id2ind[tid], id2ind[AGG_NODE]))

                adjs = pid2adjs[self.pid]
                adjs.append((id2ind[hid], id2ind[tid]))
                try:
                    for two_side in [self.subgraph_dict[hid], self.subgraph_dict[tid]]:
                        for e1, pid, e2 in two_side:
                            adjs = pid2adjs[pid]
                            adjs.append((id2ind[e1], id2ind[e2]))
                except KeyError:
                    raise DataPrepError

            # remove duplicate item in adjs
            for pid in pid2adjs:
                pid2adjs[pid] = list(set(pid2adjs[pid]))

            # build emb index
            if self.emb_id2ind:
                try:
                    emb_ind = np.array([self.emb_id2ind[k] for k in id2ind])
                except KeyError:
                    raise DataPrepError
            else:
                emb_ind = np.array([0] * len(id2ind))

            return id2ind, pid2adjs, emb_ind, 0


    @staticmethod
    def pack_graphs(graphs: List, pid2ind: OrderedDict):
        ''' convert to gnn input format (adjacency list, input emb, and prop index) '''
        if len(graphs) == 0:
            raise Exception
        edge_type = graphs[0].edge_type
        new_ind2adjs, new_emb_ind, new_prop_ind = DefaultOrderedDict(lambda: []), [], []
        acc = 0
        for g in graphs:
            for ind, adjs in g.ind2adjs.items():
                new_ind2adjs[ind].extend([(adj[0] + acc, adj[1] + acc) for adj in adjs])
            new_emb_ind.append(g.emb_ind)
            new_prop_ind.append(acc + g.prop_ind)
            acc += g.size
        if edge_type == 'one' or edge_type == 'only_property' or edge_type == 'bow':
            # could have multiple keys, must ensure the order
            adjs_li = [new_ind2adjs[pid] for pid in sorted(new_ind2adjs.keys())]
        elif edge_type == 'property':
            adjs_li = [new_ind2adjs[pid] for pid in pid2ind]  # use OrderedDict to get a list of adjs
        else:
            raise NotImplementedError
        new_emb = np.concatenate(new_emb_ind, axis=0)
        return adjs_li, new_emb, new_prop_ind


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
                 properties_as_relations: set = None,  # the property set used as relations
                 emb_id2ind: Dict[str, int] = None,
                 padding: str = PADDING,
                 edge_type: str = 'one',
                 use_cache: bool = False,
                 use_pseudo_property: bool = False,
                 use_top: bool = False):
        self.subgraph_dict = subgraph_dict
        if properties_as_relations:
            self.pid2ind = OrderedDict((pid, i) for i, pid in enumerate(properties_as_relations))
        self.edge_type = edge_type
        self.emb_id2ind = emb_id2ind
        self.padding = padding
        self.use_cache = use_cache
        self.use_pseudo_property = use_pseudo_property
        self.use_top = use_top
        self._cache = {}


    def pos_neg_filter(self, samples: List[Tuple], neg_ratio: int = 3):
        pos = [s for s in samples if s[-1] > 0]
        neg = [s for s in samples if s[-1] == 0]
        shuffle(neg)
        return pos + neg[:len(pos) * neg_ratio]


    def getids(self, getitem_result) -> set:
        ''' get the ids from the return of __getitem__ '''
        raise NotImplementedError


    def collect_ids(self) -> Dict[str, int]:
        print('collecting ids ...')
        ids = defaultdict(lambda: 0)
        for i in tqdm(range(self.__len__())):
            for id in self.getids(i):
                ids[id] += 1
        return ids


    def create_subgraph(self,
                        property_occ: Tuple[str, Tuple[Tuple[str, str]]],
                        pid2: str = None) -> PropertySubgraph:
        if self.use_cache and property_occ in self._cache:
            return self._cache[property_occ]
        pid, occs = property_occ
        # pid2 is used to avoid cheating
        sg = PropertySubgraph(
            pid, occs, self.subgraph_dict, self.emb_id2ind,
            self.padding, self.edge_type, self.use_pseudo_property, property_id2=pid2, use_top=self.use_top)
        if self.use_cache:
            self._cache[property_occ] = sg
        return sg


    def create_pair_subgraph(self,
                             property_occ1: Tuple[str, Tuple[Tuple[str, str]]],
                             property_occ2: Tuple[str, Tuple[Tuple[str, str]]]) -> PropertySubgraph:
        if self.use_cache and (property_occ1, property_occ2) in self._cache:
            return self._cache[(property_occ1, property_occ2)]
        pid1, occs1 = property_occ1
        pid2, occs2 = property_occ2
        sg = PropertySubgraph(
            pid1, occs1, self.subgraph_dict, self.emb_id2ind,
            self.padding, self.edge_type, self.use_pseudo_property,
            merge=True, property_id2=pid2, occurrences2=occs2, use_top=self.use_top)
        if self.use_cache:
            self._cache[(property_occ1, property_occ2)] = sg
        return sg


    def to(self, batch_dict: Dict[str, List], device: torch.device):
        ''' values in batch_dict might be nested lists '''
        def nested_to(li: List):
            if hasattr(li, 'to'):
                return li.to(device)
            return [nested_to(item) for item in li]
        return dict((k, nested_to(v)) for k, v in batch_dict.items() if k != 'meta')


    def item_to_str(self, item):
        fields = []
        for field in item:
            if type(field) is PropertySubgraph:
                fields.append(field.to_string())
            elif type(field) is int:
                fields.append(str(field))
        return '\t\t'.join(fields)


    def load_preprocessed_line(self, line: str):
        ''' all of the fields are PropertySubgraph except for the last one, which is label '''
        fields = line.strip().split('\t\t')
        label = int(fields[-1])
        fields = [PropertySubgraph.from_string(fields[i], edge_type=self.edge_type) for i in range(len(fields) - 1)]
        return tuple(fields) + (label,)


class PointwiseDataset(PropertyDataset):
    def __init__(self,
                 filepath: str,
                 subgraph_dict: Dict[str, List],
                 properties_as_relations: set = None,  # the property set used as relations
                 emb_id2ind: Dict[str, int] = None,
                 padding: str = PADDING,
                 edge_type: str = 'one',
                 use_cache: bool = False,
                 filter_prop: set = None,
                 keep_n_per_prop: int = None,
                 neg_ratio: int = None,
                 manager: Manager = None,
                 use_pseudo_property: bool = False,
                 use_top: bool = False):
        super(PointwiseDataset, self).__init__(
            subgraph_dict, properties_as_relations, emb_id2ind, padding,
            edge_type, use_cache, use_pseudo_property, use_top)
        print('load data from {} ...'.format(filepath))
        if filepath.endswith('.prep'):
            self.use_prep = True
            self.inst_list = load_lines(filepath)
        else:
            self.use_prep = False
            self.inst_list = read_multi_pointiwse_file(
                filepath, filter_prop=filter_prop, keep_n_per_prop=keep_n_per_prop)
            if neg_ratio:
                self.inst_list = self.pos_neg_filter(self.inst_list, neg_ratio=neg_ratio)

        # to support multiprocess, use manager to share these objects between processes and avoid copy-on-write
        # manager is quite slow because of the communication between processes
        if manager:
            assert type(emb_id2ind) is type(subgraph_dict) is multiprocessing.managers.DictProxy
            self.inst_list = manager.list(self.inst_list)
        # TODO: use numpy array or shared array to avoid copy-on-write


    def __len__(self):
        return len(self.inst_list)


    def __getitem__(self, idx):
        if self.use_prep:
            return self.load_preprocessed_line(self.inst_list[idx])
        else:
            p1o, p2o, label = self.inst_list[idx]
            return self.create_subgraph(p1o, pid2=p2o[0]), self.create_subgraph(p2o), label


    def getids(self, idx):
        p1o, p2o, label = self.inst_list[idx]
        ids = set()
        for p, occs in [p1o, p2o]:
            ids.add(p)
            for hid, tid in occs:
                ids.add(hid)
                ids.add(tid)
                for sg in [hid, tid]:
                    for h, p, t in self.subgraph_dict[sg]:
                        ids.add(h)
                        ids.add(p)
                        ids.add(t)
        return ids


    def collate_fn(self, insts: List):
        packed_sg1, packed_sg2, labels = zip(*insts)
        pid1s = [sg.pid for sg in packed_sg1]
        pid2s = [sg.pid for sg in packed_sg2]
        adjs12, emb_ind12, prop_ind12 = [], [], []
        for packed_sg in [packed_sg1, packed_sg2]:
            adjs_li, emb_ind, prop_ind = PropertySubgraph.pack_graphs(packed_sg, self.pid2ind)
            adjs_li = [AdjacencyList(node_num=len(adjs), adj_list=adjs) for adjs in adjs_li]
            emb_ind = torch.LongTensor(emb_ind)
            prop_ind = torch.LongTensor(prop_ind)
            adjs12.append(adjs_li)
            emb_ind12.append(emb_ind)
            prop_ind12.append(prop_ind)
        return {'adj': adjs12, 'emb_ind': emb_ind12, 'prop_ind': prop_ind12,
                'meta': {'pid1': pid1s, 'pid2': pid2s}}, \
               torch.LongTensor(labels)


class PointwisemergeDataset(PointwiseDataset):
    def __init__(self, *args, **kwargs):
        super(PointwisemergeDataset, self).__init__(*args, **kwargs)


    def __getitem__(self, idx):
        if self.use_prep:
            return self.load_preprocessed_line(self.inst_list[idx])
        else:
            p1o, p2o, label = self.inst_list[idx]
            return self.create_pair_subgraph(p1o, p2o), label


    def getids(self, idx):
        g, _ = self[idx]
        return g.ids


    def collate_fn(self, insts: List):
        packed_sg, labels = zip(*insts)
        pid1s = [sg.pid for sg in packed_sg]
        pid2s = [sg.pid2 for sg in packed_sg]
        adjs12, emb_ind12, prop_ind12 = [], [], []

        adjs_li, emb_ind, prop_ind = PropertySubgraph.pack_graphs(packed_sg, self.pid2ind)
        adjs_li = [AdjacencyList(node_num=len(adjs), adj_list=adjs) for adjs in adjs_li]
        emb_ind = torch.LongTensor(emb_ind)
        prop_ind = torch.LongTensor(prop_ind)
        return {'adj': [adjs_li], 'emb_ind': [emb_ind], 'prop_ind': [prop_ind],
                'meta': {'pid1': pid1s, 'pid2': pid2s}}, \
               torch.LongTensor(labels)


class NwayDataset(PropertyDataset):
    def __init__(self,
                 filepath: str,
                 subgraph_dict: Dict[str, List],
                 properties_as_relations: set = None,  # the property set used as relations
                 emb_id2ind: Dict[str, int] = None,
                 padding: str = 0,
                 edge_type: str = 'one',
                 use_cache: bool = False,
                 filter_prop: set = None,
                 keep_n_per_prop: int = None,
                 use_pseudo_property: bool = False,
                 use_top: bool = False):
        super(NwayDataset, self).__init__(
            subgraph_dict, properties_as_relations, emb_id2ind, padding,
            edge_type, use_cache, use_pseudo_property, use_top)
        print('load data from {} ...'.format(filepath))
        if filepath.endswith('.prep'):
            self.use_prep = True
            self.inst_list = load_lines(filepath)
        else:
            self.use_prep = False
            self.inst_list = read_nway_file(
                filepath, filter_prop=filter_prop, keep_n_per_prop=keep_n_per_prop)

        # TODO: use numpy array or shared array to avoid copy-on-write


    def __len__(self):
        return len(self.inst_list)


    def __getitem__(self, idx):
        if self.use_prep:
            return self.load_preprocessed_line(self.inst_list[idx])
        else:
            poccs, label = self.inst_list[idx]
            return self.create_subgraph(poccs), label


    def getids(self, idx):
        g, _ = self[idx]
        return g.ids


    def collate_fn(self, insts: List):
        packed_sg, labels = zip(*insts)
        pids = [sg.pid for sg in packed_sg]
        adjs_li, emb_ind, prop_ind = PropertySubgraph.pack_graphs(packed_sg, self.pid2ind)
        adjs_li = [AdjacencyList(node_num=len(adjs), adj_list=adjs) for adjs in adjs_li]
        emb_ind = torch.LongTensor(emb_ind)
        prop_ind = torch.LongTensor(prop_ind)
        return {'adj': [adjs_li], 'emb_ind': [emb_ind], 'prop_ind': [prop_ind],
                'meta': {'pid': pids}}, \
               torch.LongTensor(labels)


class PropertyBow:
    def __init__(self,
                 property_id: str = None,
                 occurrences: Tuple[Tuple[str, str]] = None,
                 subgraph_dict: Dict[str, List[Tuple[str, str, str]]] = None,
                 emb_id2ind: Dict[str, int] = None,
                 padding: str = PADDING,  # TODO: add padding
                 merge: bool = False,
                 property_id2: str = None,
                 occurrences2: Tuple[Tuple[str, str]] = None,
                 use_top: bool = False):
        if property_id is None:
            return
        self.pid = property_id
        self.occurrences = occurrences
        self.subgraph_dict = subgraph_dict
        self.emb_id2ind = emb_id2ind
        self.padding = padding
        self.merge = merge
        self.pid2 = property_id2
        self.occurrences2 = occurrences2
        self.use_top = use_top
        self.emb_ind, self.stat = self._to_bow()


    def normalize(self, id):
        if not self.use_top:
            return id
        if id.startswith('P'):
            if id not in TOP_PROP_IDS:
                return self.padding
        elif id.startswith('Q'):
            if id not in TOP_ENTITY_IDS:
                return self.padding
        else:
            raise Exception
        return id


    def _to_bow(self) -> Tuple[List, List]:
        id2count = defaultdict(lambda: 0)
        for hid, tid in self.occurrences:
            id2count[self.normalize(hid)] += 1
            id2count[self.normalize(tid)] += 1
            for e1, p, e2 in self.subgraph_dict[hid] + self.subgraph_dict[tid]:
                id2count[self.normalize(e1)] += 1
                id2count[self.normalize(p)] += 1
                id2count[self.normalize(e2)] += 1
        ids, counts = zip(*id2count.items())
        total_count = np.sum(counts)
        ids = [self.emb_id2ind[id] for id in ids]
        counts = [np.log(c + 1) for c in counts]  # TODO: add other stat
        return ids, counts


    def to_string(self):
        '''
        dump necessary properties to a formatted string.
        '''
        pid = self.pid
        if self.pid2 is not None:
            pid += '|' + self.pid2
        return '{pid}\t{emb_ind}\t{stat}'.format(
            pid=pid,
            emb_ind=','.join(str(i) for i in self.emb_ind),
            stat=','.join('{:.3f}'.format(i) for i in self.stat))


    @classmethod
    def from_string(cls, format_str):
        '''
        recover necessary properties from a formatted string.
        necessary properties: pid, emb_ind, stat
        '''
        empty_bow = cls()
        pid, emb_ind, stat = format_str.split('\t')
        if pid.find('|') >= 0:
            pid = pid.split('|')
            empty_bow.pid = pid[0]
            empty_bow.pid2 = pid[1]
        else:
            empty_bow.pid = pid
        empty_bow.emb_ind = list(map(int, emb_ind.split(',')))
        empty_bow.stat = list(map(float, stat.split(',')))
        return empty_bow


class BowDataset(PointwiseDataset):
    def __init__(self, *args, **kwargs):
        super(BowDataset, self).__init__(*args, **kwargs)


    def load_preprocessed_line(self, line: str):
        ''' all of the fields are PropertyBow except for the last one, which is label '''
        fields = line.strip().split('\t\t')
        label = int(fields[-1])
        fields = [PropertyBow.from_string(fields[i]) for i in range(len(fields) - 1)]
        return tuple(fields) + (label,)


    def create_bow(self, property_occ: Tuple[str, Tuple[Tuple[str, str]]]) -> PropertyBow:
        pid, occ = property_occ
        return PropertyBow(property_id=pid,
                           occurrences=occ,
                           subgraph_dict=self.subgraph_dict,
                           emb_id2ind=self.emb_id2ind,
                           padding=self.padding,
                           merge=False,
                           property_id2=None,
                           occurrences2=None,
                           use_top=self.use_top)


    def __getitem__(self, idx):
        if self.use_prep:
            return self.load_preprocessed_line(self.inst_list[idx])
        else:
            p1o, p2o, label = self.inst_list[idx]
            return self.create_bow(p1o), self.create_bow(p2o), label


    def item_to_str(self, item):
        fields = []
        for field in item:
            if type(field) is PropertyBow:
                fields.append(field.to_string())
            elif type(field) is int:
                fields.append(str(field))
        return '\t\t'.join(fields)


    def collate_fn(self, insts: List):
        bow1s, bow2s, labels = zip(*insts)
        pid1s = [b.pid for b in bow1s]
        pid2s = [b.pid for b in bow2s]
        emb_ind_li, stat_li = [], []
        for bows in [bow1s, bow2s]:
            max_len = np.max([len(b.emb_ind) for b in bows])
            emb_ind = [b.emb_ind + [0] * (max_len - len(b.emb_ind)) for b in bows]
            emb_ind_li.append(torch.LongTensor(emb_ind))
            stat = [b.stat + [0] * (max_len - len(b.stat)) for b in bows]
            stat_li.append(torch.FloatTensor(stat))
        return {'emb_ind': emb_ind_li, 'stat': stat_li,
                'meta': {'pid1': pid1s, 'pid2': pid2s}}, \
               torch.LongTensor(labels)
