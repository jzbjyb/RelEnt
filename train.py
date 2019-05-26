#!/usr/bin/env python


from typing import List, Tuple, Dict
import argparse, random, os, time
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multiprocessing import Manager
from wikiutil.data import PointwiseDataset, NwayDataset
from wikiutil.util import load_embedding, filer_embedding, load_tsv_as_dict
from wikiutil.metric import AnalogyEval, accuray
from wikiutil.property import read_prop_file, read_subgraph_file
from wikiutil.constant import AGG_NODE, AGG_PROP
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
                 method: str = 'ggnn',
                 num_edge_types: int = 1):
        super(Model, self).__init__()
        assert method in {'ggnn', 'emb'}
        self.method = method
        self.num_class = num_class
        self.num_graph = num_graph
        # get embedding
        if emb is not None:  # init from pre-trained
            vocab_size, emb_size = emb.shape
            self.emb = nn.Embedding.from_pretrained(torch.tensor(emb).float(), freeze=True)
        else:  # random init
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_ind)

        # get model
        self.emb_proj = nn.Linear(emb_size, hidden_size)
        self.gnn = GatedGraphNeuralNetwork(hidden_size=hidden_size, num_edge_types=num_edge_types,
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
                gnn = self.gnn.compute_node_representations(
                    initial_node_representation=ge, adjacency_lists=data['adj'][i])
                # TODO: combine ggnn and emb
                #ge = 0.5 * ge + 0.5 * gnn
                ge = gnn
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
        batch_dict = dataloader.dataset.to(batch_dict, device)
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
    debug = False
    keep_one_per_prop = False
    use_cache = True
    use_pseudo_property = True
    edge_type = 'property'
    num_workers = 0
    num_class_dict = {'nway': 16, 'pointwise': 1}
    num_graph_dict = {'nway': 1, 'pointwise': 2}
    Dataset = eval(args.dataset_format.capitalize() + 'Dataset')
    get_dataset_filepath = lambda split: os.path.join(args.dataset_dir, split + '.' + args.dataset_format)

    # load subgraph
    subgraph_dict = read_subgraph_file(args.subgraph_file)

    # filter emb and get all the properties used by dry run datasets
    if args.filter_emb:
        all_ids = defaultdict(lambda: 0)
        for split in ['train', 'dev', 'test']:
            ds = Dataset(get_dataset_filepath(split), subgraph_dict, edge_type='one')
            for k, v in ds.collect_ids().items():
                all_ids[k] += v

        print('#ids {}'.format(len(all_ids)))
        all_properties = [(k, v) for k, v in all_ids.items() if k.startswith('P')]
        print('#properties {}'.format(len(all_properties)))

        with open(os.path.join(args.dataset_dir, 'pid2ind.tsv'), 'w') as fout:
            for i, (pid, count) in enumerate(sorted(all_properties, key=lambda x: -x[1])):
                fout.write('{}\t{}\t{}\n'.format(pid, i, count))
        filer_embedding(args.emb_file, 'data/test.emb', set(all_ids.keys()))
        exit(1)

    # load embedding
    id2ind, emb = load_embedding(args.emb_file, debug=debug, emb_size=200) if args.emb_file else (None, None)
    #properties_as_relations = {'P31', 'P21', 'P527', 'P17'}  # for debug
    properties_as_relations = load_tsv_as_dict(os.path.join(args.dataset_dir, 'pid2ind.tsv'), valuefunc=int)
    # add agg property and node
    properties_as_relations[AGG_PROP] = len(properties_as_relations)
    id2ind[AGG_NODE] = len(id2ind)
    emb = np.concatenate([emb, np.zeros((1, emb.shape[1]), dtype=np.float32)], axis=0)
    # set num_edge_types
    num_edge_types = 1 if edge_type == 'one' else len(properties_as_relations)

    # load data
    dataset_params = {
        'subgraph_dict': subgraph_dict,
        'properties_as_relations': properties_as_relations,
        'id2ind': id2ind,
        'edge_type': edge_type,
        'keep_one_per_prop': keep_one_per_prop,
        'use_cache': use_cache,
        'use_pseudo_property': use_pseudo_property
    }
    get_dataloader = lambda ds, shuffle: \
        DataLoader(ds, batch_size=64, shuffle=shuffle,
                   num_workers=num_workers, collate_fn=ds.collate_fn)

    train_data = Dataset(get_dataset_filepath('train'), **dataset_params)
    train_dataloader = get_dataloader(train_data, True)
    dev_data = Dataset(get_dataset_filepath('dev'), **dataset_params)
    dev_dataloader = get_dataloader(dev_data, False)
    test_data = Dataset(get_dataset_filepath('test'), **dataset_params)
    test_dataloader = get_dataloader(test_data, False)

    # config model, optimizer and evaluation
    model = Model(num_class=num_class_dict[args.dataset_format],
                  num_graph=num_graph_dict[args.dataset_format],
                  emb=emb,
                  hidden_size=64,
                  method='ggnn',
                  num_edge_types=num_edge_types)
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
