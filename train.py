#!/usr/bin/env python


from typing import List, Tuple, Dict
import argparse, random, os, time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multiprocessing import Manager
from wikiutil.data import PointwiseDataset, NwayDataset
from wikiutil.util import load_embedding, filer_embedding
from wikiutil.metric import AnalogyEval, accuray
from wikiutil.property import read_prop_file, read_subgraph_file
from analogy.ggnn import GatedGraphNeuralNetwork


class Model(nn.Module):
    def __init__(self,
                 num_class: int = 1,  # number of classes for prediction
                 num_graph: int = 2,  # number of graphs in data dict
                 emb_size: int = 100,
                 vocab_size: int = None,
                 padding_ind: int = 0,  # TODO: add padding index
                 emb: np.ndarray = None,
                 hidden_size: int = 64,
                 method: str = 'ggnn'):
        super(Model, self).__init__()
        assert method in {'ggnn', 'emb'}
        self.method = method
        self.num_class = num_class
        self.num_graph = num_graph
        # get embedding
        if emb is not None:  # init from pre-trained
            vocab_size, emb_size = emb.shape
            self.emb = nn.Embedding.from_pretrained(torch.tensor(emb), freeze=True)
        else:  # random init
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_ind)

        # get model
        self.emb_proj = nn.Linear(emb_size, hidden_size)
        self.gnn = GatedGraphNeuralNetwork(hidden_size=hidden_size, num_edge_types=1,
                                           layer_timesteps=[3, 3], residual_connections={})
        self.cla = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size * num_graph, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, num_class)
        )


    def forward(self, data: Dict, labels: torch.LongTensor):
        ge_list = []
        for i in range(self.num_graph):
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
        ge = torch.cat(ge_list, -1)
        if self.num_class == 1:
            # binary classification loss
            logits = torch.sigmoid(self.cla(ge))
            labels = labels.float()
            loss = nn.BCELoss()(logits.squeeze(), labels)
        else:
            # cross-entropy loss
            logits = self.cla(ge)
            loss = nn.CrossEntropyLoss()(logits, labels)
        return logits, loss


def one_epoch(args, split, dataloader, optimizer, device, show_progress=True):
    loss_li, pred_li = [], []
    if split == 'train':
        model.train()
        iter = tqdm(dataloader, disable=not show_progress)
    else:
        model.eval()
        iter = tqdm(dataloader, disable=not show_progress)
    for batch in iter:
        batch_dict, label = batch
        batch_dict = dict((k, [t.to(device) for t in batch_dict[k]]) for k in batch_dict if k != 'meta')
        label = label.to(device)
        logits, loss = model(batch_dict, label)

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_li.append(loss.item())
        if args.dataset_format == 'pointwise':
            pred_li.extend([(pid1, pid2, logits[i].item())
                            for i, (pid1, pid2) in
                            enumerate(zip(batch[0]['meta']['pid1'], batch[0]['meta']['pid2']))])
        else:
            pred_li.extend([(pid, logits[i].detach().cpu().numpy(), label[i].item())
                            for i, pid in enumerate(batch[0]['meta']['pid'])])

    return pred_li, loss_li


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train analogy model')
    parser.add_argument('--dataset_dir', type=str, required=True, help='dataset dir')
    parser.add_argument('--dataset_format', type=str, choices=['pointwise', 'nway'],
                        default='pointwise', help='dataset format')
    parser.add_argument('--subgraph_file', type=str, required=True, help='entity subgraph file')
    parser.add_argument('--subprop_file', type=str, required=True, help='subprop file')
    parser.add_argument('--emb_file', type=str, default=None, help='embedding file')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--filter_emb', action='store_true')
    parser.add_argument('--patience', type=int, default=0, help='number of epoch before running out of patience')
    args = parser.parse_args()

    random.seed(2019)
    np.random.seed(2019)
    torch.manual_seed(2019)

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # get dataset class and configs
    keep_one_per_prop = False
    use_cache = True
    num_worker = 0
    num_class_dict = {'nway': 16, 'pointwise': 1}
    num_graph_dict = {'nway': 1, 'pointwise': 2}
    Dataset = eval(args.dataset_format.capitalize() + 'Dataset')
    get_dataset_filepath = lambda split: os.path.join(args.dataset_dir, split + '.' + args.dataset_format)

    # load subgraph
    subgraph_dict = read_subgraph_file(args.subgraph_file)

    # filter emb by dry run datasets
    if args.filter_emb:
        all_ids = set()
        for split in ['train', 'dev', 'test']:
            ds = Dataset(get_dataset_filepath(split), subgraph_dict, edge_type='one')
            all_ids.update(ds.collect_ids())
        print('#ids {}'.format(len(all_ids)))
        filer_embedding(args.emb_file, 'data/test.emb', all_ids)
        exit(1)

    # load embedding
    id2ind, emb = load_embedding(args.emb_file, debug=False, emb_size=200) if args.emb_file else (None, None)

    # load data
    train_data = Dataset(get_dataset_filepath('train'), subgraph_dict,
                         id2ind=id2ind, edge_type='one', keep_one_per_prop=keep_one_per_prop,
                         use_cache=use_cache)
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True,
                                  num_workers=num_worker, collate_fn=train_data.collate_fn)
    dev_data = Dataset(get_dataset_filepath('dev'), subgraph_dict,
                       id2ind=id2ind, edge_type='one', keep_one_per_prop=keep_one_per_prop,
                       use_cache=use_cache)
    dev_dataloader = DataLoader(dev_data, batch_size=128, shuffle=False,
                                num_workers=num_worker, collate_fn=dev_data.collate_fn)
    test_data = Dataset(get_dataset_filepath('test'), subgraph_dict,
                        id2ind=id2ind, edge_type='one', keep_one_per_prop=keep_one_per_prop,
                        use_cache=use_cache)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False,
                                 num_workers=num_worker, collate_fn=test_data.collate_fn)

    # config model, optimizer and evaluation
    model = Model(num_class=num_class_dict[args.dataset_format],
                  num_graph=num_graph_dict[args.dataset_format],
                  emb=emb,
                  hidden_size=64,
                  method='ggnn')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 1e-3 for emb and 1e-4 for ggnn
    '''
    train_prop_set = set(read_prop_file(os.path.join(args.dataset_dir, 'train.prop')))
    train_metirc = AnalogyEval(args.subprop_file, method='parent', metric='auc_map',
                               reduction='property', prop_set=train_prop_set)
    dev_prop_set = set(read_prop_file(os.path.join(args.dataset_dir, 'dev.prop')))
    dev_metric = AnalogyEval(args.subprop_file, method='parent', metric='auc_map',
                             reduction='property', prop_set=dev_prop_set, debug=False)
    '''

    # train and test
    show_progress = False
    pat, best_dev_metric = 0, 0
    for epoch in range(300):
        # init performance
        if epoch == 0:
            dev_pred, dev_loss = one_epoch(args, 'dev', dev_dataloader, optimizer,
                                           device=device, show_progress=show_progress)
            print('init')
            #print(np.mean(dev_loss), dev_metric.eval(dev_pred))
            print(np.mean(dev_loss), accuray(dev_pred))

        # train, dev, test
        train_pred, train_loss = one_epoch(args, 'train', train_dataloader, optimizer,
                                           device=device, show_progress=show_progress)
        dev_pred, dev_loss = one_epoch(args, 'dev', dev_dataloader, optimizer,
                                       device=device, show_progress=show_progress)
        test_pred, test_loss = one_epoch(args, 'test', test_dataloader, optimizer,
                                         device=device, show_progress=show_progress)

        # evaluate
        train_metric, dev_metric, test_metric = accuray(train_pred), accuray(dev_pred), accuray(test_pred)
        print('epoch {:4d}\ttr_loss: {:>.3f}\tdev_loss: {:>.3f}\tte_loss: {:>.3f}'.format(
            epoch + 1, np.mean(train_loss), np.mean(dev_loss), np.mean(test_loss)), end='')
        print('\t\ttr_acc: {:>.3f}\tdev_acc: {:>.3f}\ttest_acc: {:>.3f}'.format(
            train_metric, dev_metric, test_metric))
        '''
        print('accuracy')
        print(train_metirc.eval_by('property', 'accuracy', train_pred),
              dev_metric.eval_by('property', 'accuracy', dev_pred))
        print('correct_position')
        print(train_metirc.eval_by('property', 'correct_position', train_pred),
              dev_metric.eval_by('property', 'correct_position', dev_pred))
        '''

        # early stop
        if args.patience:
            if dev_metric < best_dev_metric:
                pat += 1
                if pat >= args.patience:
                    print('run out of patience')
                    break
            else:
                pat = 0
                best_dev_metric = dev_metric
