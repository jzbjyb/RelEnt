from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from itertools import combinations

from wikiutil.property import PropertySubtree, read_subprop_file, get_all_subtree
from .gcn import MyGCNConv


class EmbGnnModel(nn.Module):
    def __init__(self, feat_size, hidden_size, num_class, dropout=0.0, method='gcn1_diag'):
        super(EmbGnnModel, self).__init__()
        self.num_class = num_class
        assert method in {'none', 'gcn1_diag', 'gcn1', 'gcn2', 'gat1'}
        self.method = method
        if method == 'gcn1_diag':
            self.gcn1 = MyGCNConv(feat_size, feat_size, improved=False, learnable=True)
        elif method == 'gcn1':
            self.gcn1 = GCNConv(feat_size, feat_size, improved=False)
        self.gat1 = GATConv(feat_size, feat_size)
        self.ff = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.gcn2 = GCNConv(hidden_size, hidden_size, improved=False)
        self.pred_ff = nn.Linear(hidden_size, num_class)


    def forward(self,
                feature: torch.FloatTensor,  # SHAPE: (num_nodes, feat_size)
                adj: torch.LongTensor,  # SHAPE: (2, num_edges)
                label: torch.LongTensor,  # SHAPE: (num_nodes,)
                train_mask: torch.LongTensor,  # SHAPE: (num_nodes,)
                dev_mask: torch.LongTensor,  # SHAPE: (num_nodes,)
                test_mask: torch.LongTensor):  # SHAPE: (num_nodes,)
        if self.method == 'gcn1' or self.method == 'gcn1_diag':
            feature = self.gcn1(feature, adj)
        elif self.method == 'gat1':
            feature = self.gat1(feature, adj)
        pred_repr = self.ff(feature)
        if self.method == 'gcn2':
            pred_repr = self.gcn2(pred_repr, adj)

        # SHAPE: (num_nodes, num_class)
        logits = self.pred_ff(pred_repr)
        loss = nn.CrossEntropyLoss(reduction='none')(logits, label)

        train_mask = train_mask.eq(1)
        dev_mask = dev_mask.eq(1)
        test_mask = test_mask.eq(1)

        train_logits = torch.masked_select(logits, train_mask.unsqueeze(-1)).view(-1, self.num_class)
        dev_logits = torch.masked_select(logits, dev_mask.unsqueeze(-1)).view(-1, self.num_class)
        test_logits = torch.masked_select(logits, test_mask.unsqueeze(-1)).view(-1, self.num_class)

        train_loss = torch.masked_select(loss, train_mask).mean()
        dev_loss = torch.masked_select(loss, dev_mask).mean()
        test_loss = torch.masked_select(loss, test_mask).mean()

        return train_logits, dev_logits, test_logits, train_loss, dev_loss, test_loss


def load_seealso(seealso_file, subprop_file, use_split=False,
                 train_pids=None, dev_pids=None, test_pids=None):
    # load subprop
    subprops = read_subprop_file(subprop_file)
    subtrees, isolate = get_all_subtree(subprops)

    pid2nei_split: Dict[str, set] = defaultdict(set)
    pid2newpid: Dict[str, str] = {}
    if use_split:
        pid2split = {}
        for split, pids in enumerate([train_pids, dev_pids, test_pids]):
            for pid in pids:
                if pid in pid2split:
                    raise Exception('duplicate pids')
                pid2split[pid] = split
        # special cases for split tier
        for subtree in subtrees:
            for pid, childs in subtree.traverse_each_tier():
                if not PropertySubtree.is_split(pid, childs):
                    continue
                rawpid = pid.split('_')[0]
                if pid != rawpid:
                    pid2newpid[rawpid] = pid
                for c1, c2 in combinations(childs, 2):
                    if c1 in pid2split and c2 in pid2split and pid2split[c1] == pid2split[c2]:
                        # in the same split
                        pid2nei_split[c1].add(c2)
                        pid2nei_split[c2].add(c1)

    # load see also
    pid2nei: Dict[str, set] = defaultdict(set)
    with open(seealso_file, 'r') as fin:
        for l in fin:
            l = l.strip()
            if l == '':
                continue
            h, _, t = l.split('\t')
            if h in pid2newpid:
                h = pid2newpid[h]
            if t in pid2newpid:
                t = pid2newpid[t]
            pid2nei[h].add(t)

    if use_split:
        pid2nei.update(pid2nei_split)

    return dict(pid2nei)


def build_graph(train_samples, dev_samples, test_samples, seealso_file, subprop_file,
                emb_id2ind, emb):
    pid2ind: Dict[str, int] = defaultdict(lambda: len(pid2ind))
    # build features and labels
    feats: List[np.ndarry] = []
    labels: List[int] = []
    mask: List[int] = []
    pids: List[int] = []
    for split, samples in enumerate([train_samples, dev_samples, test_samples]):
        for (pid, pocc), plabel in samples:
            if pid in pid2ind:
                raise Exception('duplicate pids')
            pind = pid2ind[pid]
            feat = []
            for h, t in pocc:
                feat.append(np.concatenate([emb[emb_id2ind[h]], emb[emb_id2ind[t]]]))
            feat = np.mean(np.array(feat), 0)
            feats.append(feat)
            labels.append(plabel)
            mask.append(split)
            pids.append(pid)

    pid2ind = dict(pid2ind)
    feats = np.array(feats)
    labels = np.array(labels)
    mask = np.array(mask)
    pids = np.array(pids)
    train_mask = mask == 0
    dev_mask = mask == 1
    test_mask = mask == 2

    train_pids = pids[train_mask]
    dev_pids = pids[dev_mask]
    test_pids = pids[test_mask]

    # build neighbour
    pid2nei = load_seealso(seealso_file, subprop_file, use_split=True,
                           train_pids=train_pids, dev_pids=dev_pids, test_pids=test_pids)

    # build adj
    adj: List[Tuple[int, int]] = set()
    for pid, nei in pid2nei.items():
        for n in nei:
            if pid not in pid2ind or n not in pid2ind:
                continue
            # undirectional
            adj.add((pid2ind[pid], pid2ind[n]))
            adj.add((pid2ind[n], pid2ind[pid]))
    print('#edges {}'.format(len(adj)))
    adj = np.array(list(adj), dtype=int).T

    tensor_data = {
        'feature': torch.tensor(feats).float(),
        'adj': torch.tensor(adj).long(),
        'label': torch.tensor(labels).long(),
        'train_mask': torch.tensor(train_mask).long(),
        'dev_mask': torch.tensor(dev_mask).long(),
        'test_mask': torch.tensor(test_mask).long(),
    }

    meta_data = {
        'train_pids': train_pids,
        'dev_pids': dev_pids,
        'test_pids': test_pids,
        'train_labels': labels[train_mask],
        'dev_labels': labels[dev_mask],
        'test_labels': labels[test_mask]
    }

    return tensor_data, meta_data
