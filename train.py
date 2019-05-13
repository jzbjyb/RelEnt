#!/usr/bin/env python


from typing import List, Tuple
import argparse, random, os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from wikiutil.data import PointwiseDataLoader, PropertySubgraph, filer_embedding
from wikiutil.metric import AnalogyEval
from analogy.ggnn import GatedGraphNeuralNetwork, AdjacencyList


def pointwise_batch_to_tensor(batch: List[PropertySubgraph], device) \
        -> Tuple[List, torch.LongTensor]:
    ''' python data to pytorch data '''
    data = []
    labels = []
    for sg1, sg2, label in batch:
        e1 = torch.tensor(sg1.emb, dtype=torch.float).to(device)
        e2 = torch.tensor(sg2.emb, dtype=torch.float).to(device)
        adjs1 = AdjacencyList(node_num=len(sg1.adjs), adj_list=sg1.adjs, device=device)
        adjs2 = AdjacencyList(node_num=len(sg2.adjs), adj_list=sg2.adjs, device=device)
        data.append(((e1, adjs1), (e2, adjs2)))
        labels.append(label)
    return data, torch.LongTensor(labels)


class ModelWrapper(nn.Module):
    def __init__(self, emb_size, hidden_size=64, method='ggnn'):
        super(ModelWrapper, self).__init__()
        assert method in {'ggnn', 'emb'}
        self.method = method
        self.emb_proj = nn.Linear(emb_size, hidden_size)
        self.gnn = GatedGraphNeuralNetwork(hidden_size=hidden_size, num_edge_types=1,
                                           layer_timesteps=[3, 3], residual_connections={})
        self.binary_cla = nn.Linear(hidden_size * 2, 1)


    def forward(self, data: List, labels: torch.LongTensor):
        g1s, g2s = [], []
        for (g1e, g1_adj), (g2e, g2_adj) in data:
            g1e = self.emb_proj(g1e)
            g2e = self.emb_proj(g2e)
            if self.method == 'ggnn':
                g1pe = self.gnn.compute_node_representations(
                    initial_node_representation=g1e, adjacency_lists=[g1_adj])[0]
                g2pe = self.gnn.compute_node_representations(
                    initial_node_representation=g2e, adjacency_lists=[g2_adj])[0]
            elif self.method == 'emb':
                g1pe = g1e[0]
                g2pe = g2e[0]
            g1s.append(g1pe)
            g2s.append(g2pe)
        g1s = torch.stack(g1s, 0)
        g2s = torch.stack(g2s, 0)
        logits = torch.sigmoid(self.binary_cla(torch.cat([g1s, g2s], -1)))
        labels = labels.float()
        loss = nn.BCELoss()(logits.squeeze(), labels)
        return logits, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train analogy model')
    parser.add_argument('--dataset_dir', type=str, required=True, help='dataset dir')
    parser.add_argument('--subgraph_file', type=str, required=True, help='entity subgraph file')
    parser.add_argument('--subprop_file', type=str, required=True, help='subprop file')
    parser.add_argument('--emb_file', type=str, default=None, help='embedding file')
    parser.add_argument('--no_cuda', action='store_true')
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
                                     emb_file=args.emb_file,
                                     edge_type='one')

    # filter emb
    print('#ids {}'.format(len(dataloader.all_ids)))
    #filer_embedding(args.emb_file, 'data/test.emb', dataloader.all_ids)

    model = ModelWrapper(emb_size=200, hidden_size=32, method='emb')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    metric = AnalogyEval(args.subprop_file, method='accuracy', reduction='property')

    for epoch in range(20):
        train_loss = []
        model.train()
        for batch in tqdm(dataloader.batch_iter('train', batch_size=64, batch_per_epoch=200, repeat=True)):
            logits, loss = model(*pointwise_batch_to_tensor(batch, device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        dev_loss, dev_result = [], []
        model.eval()
        for batch in tqdm(dataloader.batch_iter('dev', batch_size=64, restart=True)):
            logits, loss = model(*pointwise_batch_to_tensor(batch, device))
            dev_loss.append(loss.item())
            dev_result.extend([(g1.pid, g2.pid, logits[i].item()) for i, (g1, g2, label) in enumerate(batch)])

        print(np.mean(train_loss), np.mean(dev_loss), metric.eval(dev_result))
