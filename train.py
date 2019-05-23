#!/usr/bin/env python


from typing import List, Tuple, Dict
import argparse, random, os, time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multiprocessing import Manager
from wikiutil.data import PointwiseDataLoader, PointwiseDataset
from wikiutil.util import load_embedding, filer_embedding
from wikiutil.metric import AnalogyEval
from wikiutil.property import read_prop_file, read_subgraph_file
from analogy.ggnn import GatedGraphNeuralNetwork


class ModelWrapper(nn.Module):
    def __init__(self,
                 emb_size: int,
                 vocab_size: int = None,
                 padding_ind: int = 0,  # TODO: add padding index
                 emb: np.ndarray = None,
                 hidden_size: int = 64,
                 method: str = 'ggnn'):
        super(ModelWrapper, self).__init__()
        assert method in {'ggnn', 'emb'}
        self.method = method
        # get embedding
        if emb is not None:  # init from pre-trained
            self.emb = nn.Embedding.from_pretrained(torch.tensor(emb), freeze=True)
        else:  # random init
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_ind)

        # get model
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
            ge = self.emb_proj(self.emb(data['emb_ind'][i]))
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


def one_epoch(split, dataloader, optimizer, device):
    loss_li, pred_li = [], []
    if split == 'train':
        model.train()
        #iter = tqdm(dataloader.batch_iter(split, batch_size=1024, batch_per_epoch=100, repeat=True))
        iter = tqdm(dataloader)
    else:
        model.eval()
        #iter = tqdm(dataloader.batch_iter(split, batch_size=1024, restart=True))
        iter = tqdm(dataloader)
    for batch in iter:
        #logits, loss = model(*pointwise_batch_to_tensor(batch, device))
        batch_dict, label = batch
        batch_dict = dict((k, [t.to(device) for t in batch_dict[k]]) for k in batch_dict if k != 'meta')
        label = label.to(device)
        logits, loss = model(batch_dict, label)

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_li.append(loss.item())
        #pred_li.extend([(g1.pid, g2.pid, logits[i].item()) for i, (g1, g2, label) in enumerate(batch)])
        pred_li.extend([(pid1, pid2, logits[i].item())
                        for i, (pid1, pid2) in
                        enumerate(zip(batch[0]['meta']['pid1'], batch[0]['meta']['pid2']))])
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

    # filter emb
    if args.filter_emb:
        dataloader = PointwiseDataLoader(os.path.join(args.dataset_dir, 'train.pointwise'),
                                         os.path.join(args.dataset_dir, 'dev.pointwise'),
                                         os.path.join(args.dataset_dir, 'test.pointwise'),
                                         args.subgraph_file,
                                         emb_file=args.emb_file if not args.filter_emb else None,
                                         emb_size=None if not args.filter_emb else 32,
                                         edge_type='one')
        print('#ids {}'.format(len(dataloader.all_ids)))
        filer_embedding(args.emb_file, 'data/test.emb', dataloader.all_ids)
        exit(1)

    # load subgraph and embedding
    subgraph_dict = read_subgraph_file(args.subgraph_file)
    id2ind, emb = load_embedding(args.emb_file, debug=True, emb_size=200) if args.emb_file else (None, None)
    '''
    # TODO debug
    class emb_dict_cls():
        def __getitem__(self, item):
            return [0.1] * 200
    emb_dict = emb_dict_cls()
    '''

    # load data
    train_data = PointwiseDataset(os.path.join(args.dataset_dir, 'train.pointwise'),
                                  subgraph_dict,
                                  id2ind=id2ind,
                                  edge_type='one',
                                  keep_one_per_prop=False)
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True,
                                  num_workers=8, collate_fn=train_data.collate_fn)
    dev_data = PointwiseDataset(os.path.join(args.dataset_dir, 'dev.pointwise'),
                                subgraph_dict,
                                id2ind=id2ind,
                                edge_type='one',
                                keep_one_per_prop=False)
    dev_dataloader = DataLoader(dev_data, batch_size=128, shuffle=False,
                                num_workers=8, collate_fn=dev_data.collate_fn)

    # config model, optimizer and evaluation
    model = ModelWrapper(emb_size=200, emb=emb, hidden_size=64, method='ggnn')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_prop_set = set(read_prop_file(os.path.join(args.dataset_dir, 'train.prop')))
    train_metirc = AnalogyEval(args.subprop_file, method='auc_map',
                               reduction='property', prop_set=train_prop_set)
    dev_prop_set = set(read_prop_file(os.path.join(args.dataset_dir, 'dev.prop')))
    dev_metric = AnalogyEval(args.subprop_file, method='auc_map',
                             reduction='property', prop_set=dev_prop_set, debug=False)

    # train and evaluate
    for epoch in range(100):

        print('epoch {}'.format(epoch + 1))
        if epoch == 0:
            dev_pred, dev_loss = one_epoch('dev', dev_dataloader, optimizer, device=device)
            print('init')
            print(np.mean(dev_loss), dev_metric.eval(dev_pred))

        train_pred, train_loss = one_epoch('train', train_dataloader, optimizer, device=device)
        dev_pred, dev_loss = one_epoch('dev', dev_dataloader, optimizer, device=device)

        print('tr_loss: {:>.3f}\tdev_loss: {:>.3f}'.format(np.mean(train_loss), np.mean(dev_loss)))
        print('accuracy')
        print(train_metirc.eval_by('property', 'accuracy', train_pred),
              dev_metric.eval_by('property', 'accuracy', dev_pred))
        print('ranking')
        print(train_metirc.eval_by('property', 'auc_map', train_pred),
              dev_metric.eval_by('property', 'auc_map', dev_pred))
