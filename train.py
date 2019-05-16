#!/usr/bin/env python


from typing import List, Tuple, Dict
import argparse, random, os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from wikiutil.data import PointwiseDataLoader, PropertySubgraph, filer_embedding
from wikiutil.metric import AnalogyEval
from wikiutil.property import read_prop_file
from analogy.ggnn import GatedGraphNeuralNetwork, AdjacencyList


def pointwise_batch_to_tensor(batch: List[PropertySubgraph], device) \
        -> Tuple[Dict, torch.LongTensor]:
    ''' python data to pytorch data '''
    packed_sg1, packed_sg2, labels = zip(*batch)
    adjs12, e12, prop_ind12 = [], [], []
    for packed_sg in [packed_sg1, packed_sg2]:
        adjs, e, prop_ind = PropertySubgraph.pack_graphs(packed_sg)
        adjs = AdjacencyList(node_num=len(adjs), adj_list=adjs, device=device)
        e = torch.tensor(e, dtype=torch.float).to(device)
        prop_ind = torch.tensor(prop_ind).to(device)
        adjs12.append(adjs)
        e12.append(e)
        prop_ind12.append(prop_ind)
    return {'adj': adjs12, 'emb': e12, 'prop_ind': prop_ind12}, \
           torch.LongTensor(labels).to(device)


class ModelWrapper(nn.Module):
    def __init__(self, emb_size, hidden_size=64, method='ggnn'):
        super(ModelWrapper, self).__init__()
        assert method in {'ggnn', 'emb'}
        self.method = method
        self.emb_proj = nn.Linear(emb_size, hidden_size)
        self.gnn = GatedGraphNeuralNetwork(hidden_size=hidden_size, num_edge_types=1,
                                           layer_timesteps=[3, 3], residual_connections={})
        self.binary_cla = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, 1)
        )


    def forward(self, data: Dict, labels: torch.LongTensor):
        ge_list = []
        for i in range(2):
            # get representation
            ge = self.emb_proj(data['emb'][i])
            if self.method == 'ggnn':
                ge = self.gnn.compute_node_representations(
                    initial_node_representation=ge, adjacency_lists=[data['adj'][i]])
            # select
            # SHAPE: (batch_size, emb_size)
            ge = torch.index_select(ge, 0, data['prop_ind'][i])
            ge_list.append(ge)
        # match
        logits = torch.sigmoid(self.binary_cla(torch.cat(ge_list, -1)))
        labels = labels.float()
        loss = nn.BCELoss()(logits.squeeze(), labels)
        return logits, loss


def one_epoch(split, dataloader, optimizer):
    loss_li, pred_li = [], []
    if split == 'train':
        model.train()
        #iter = tqdm(dataloader.batch_iter(split, batch_size=64, batch_per_epoch=200, repeat=True))
        iter = tqdm(dataloader.batch_iter(split, batch_size=1024, restart=True))
    else:
        iter = tqdm(dataloader.batch_iter(split, batch_size=1024, restart=True))
        model.eval()
    for batch in iter:
        logits, loss = model(*pointwise_batch_to_tensor(batch, device))
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_li.append(loss.item())
        pred_li.extend([(g1.pid, g2.pid, logits[i].item()) for i, (g1, g2, label) in enumerate(batch)])
    return pred_li, loss_li


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train analogy model')
    parser.add_argument('--dataset_dir', type=str, required=True, help='dataset dir')
    parser.add_argument('--subgraph_file', type=str, required=True, help='entity subgraph file')
    parser.add_argument('--subprop_file', type=str, required=True, help='subprop file')
    parser.add_argument('--emb_file', type=str, default=None, help='embedding file')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--filter_emb', action='store_true')
    args = parser.parse_args()

    random.seed(2019)
    np.random.seed(2019)
    torch.manual_seed(2019)

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    dataloader = PointwiseDataLoader(os.path.join(args.dataset_dir, 'train.pointwise'),
                                     os.path.join(args.dataset_dir, 'dev.pointwise'),
                                     os.path.join(args.dataset_dir, 'test.pointwise'),
                                     args.subgraph_file,
                                     emb_file=args.emb_file if not args.filter_emb else None,
                                     emb_size=None if not args.filter_emb else 32,
                                     edge_type='one')

    # filter emb
    print('#ids {}'.format(len(dataloader.all_ids)))
    if args.filter_emb:
        filer_embedding(args.emb_file, 'data/test.emb', dataloader.all_ids)
        exit(1)

    model = ModelWrapper(emb_size=200, hidden_size=64, method='emb')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_prop_set = set(read_prop_file(os.path.join(args.dataset_dir, 'train.prop')))
    train_metirc = AnalogyEval(args.subprop_file, method='auc_map', reduction='property', prop_set=train_prop_set)
    dev_prop_set = set(read_prop_file(os.path.join(args.dataset_dir, 'dev.prop')))
    dev_metric = AnalogyEval(args.subprop_file, method='auc_map',
                             reduction='property', prop_set=dev_prop_set, debug=True)

    for epoch in range(20):

        print('epoch {}'.format(epoch + 1))
        if epoch == 0:
            dev_pred, dev_loss = one_epoch('dev', dataloader, optimizer)
            print('init')
            print(np.mean(dev_loss), dev_metric.eval(dev_pred))

        train_pred, train_loss = one_epoch('train', dataloader, optimizer)
        dev_pred, dev_loss = one_epoch('dev', dataloader, optimizer)

        #print('{:>.3f}\t{:>.3f}\t{:>.3f}\t{:>.3f}'.format(
        #    np.mean(train_loss), np.mean(dev_loss), train_metirc.eval(train_pred), dev_metric.eval(dev_pred)))
        print('tr_loss: {:>.3f}\tdev_loss: {:>.3f}'.format(np.mean(train_loss), np.mean(dev_loss)))
        print(train_metirc.eval(train_pred), dev_metric.eval(dev_pred))
