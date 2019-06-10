#!/usr/bin/env python


from typing import List, Tuple, Dict
import argparse, random, os, time
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from wikiutil.data import PointwiseDataset, NwayDataset, PointwisemergeDataset, BowDataset
from wikiutil.util import filer_embedding, load_tsv_as_dict, read_embeddings_from_text_file
from wikiutil.metric import AnalogyEval, accuracy_nway, accuracy_pointwise, rank_to_csv
from wikiutil.property import read_prop_file, read_subgraph_file, read_subprop_file
from wikiutil.constant import AGG_NODE, AGG_PROP, PADDING
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
                 num_edge_types: int = 1,
                 layer_timesteps: List[int] = [3, 3],
                 match: str = 'concat'):
        super(Model, self).__init__()
        assert method in {'ggnn', 'emb', 'cosine', 'bow'}
        self.method = method
        assert match in {'concat', 'cosine'}
        # 'concat': embeddings of two relations are concatenated to make prediction
        # 'cosine': embeddings of two relations are compared by cosine similarity
        self.match = match
        self.num_class = num_class
        self.num_graph = num_graph
        # get embedding
        self.padding_ind = padding_ind
        if emb is not None:  # init from pre-trained
            vocab_size, emb_size = emb.shape
            self.emb = nn.Embedding.from_pretrained(
                torch.tensor(emb).float(), freeze=True, padding_idx=padding_ind)
        else:  # random init
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_ind)

        # get model
        self.emb_proj = nn.Linear(emb_size, hidden_size)
        self.gnn = GatedGraphNeuralNetwork(hidden_size=hidden_size, num_edge_types=num_edge_types,
                                           layer_timesteps=layer_timesteps, residual_connections={})
        self.cosine_bias = nn.Parameter(torch.zeros(1))
        self.cosine_cla = lambda x: x + self.cosine_bias

        self.cla = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(hidden_size * num_graph, 16),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(16, num_class)
        )

    def forward(self, data: Dict, labels: torch.LongTensor):
        ge_list = []
        for i in range(self.num_graph):
            # get representation
            ge = self.emb(data['emb_ind'][i])
            if self.method == 'bow':
                # average using stat
                # SHAPE: (batch_size, emb_size)
                ge = (data['stat'][i].unsqueeze(-1) * ge).sum(1)  # TODO: add bias?
                #ge /= data['emb_ind'][i].ne(self.padding_ind).sum(-1).float().unsqueeze(-1)
                ge = F.relu(ge)
            else:
                if self.method != 'cosine':
                    ge = self.emb_proj(ge)
                if self.method == 'ggnn':
                    gnn = self.gnn.compute_node_representations(
                        initial_node_representation=ge, adjacency_lists=data['adj'][i])
                    # TODO: combine ggnn and emb
                    #ge = 0.5 * ge + 0.5 * gnn
                    ge = gnn
                # SHAPE: (batch_size, emb_size)
                ge = torch.index_select(ge, 0, data['prop_ind'][i])  # select property emb
            ge_list.append(ge)
        # match
        if self.match == 'concat':
            ge = torch.cat(ge_list, -1)
            ge = self.cla(ge)
        elif self.match == 'cosine':
            ge = self.cosine_cla(F.cosine_similarity(ge_list[0], ge_list[1]).unsqueeze(-1))

        if self.num_class == 1:
            # binary classification loss
            logits = torch.sigmoid(ge)
            labels = labels.float()
            loss = nn.BCELoss()(logits.squeeze(), labels)
        else:
            # cross-entropy loss
            logits = ge
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

        if split == 'train' and loss.requires_grad is True:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_li.append(loss.item())
        if args.dataset_format == 'pointwise' or \
                args.dataset_format == 'pointwisemerge' or \
                args.dataset_format == 'bow':
            pred_li.extend([(pid1, pid2, logits[i].item(), label[i].item())
                            for i, (pid1, pid2) in
                            enumerate(zip(batch[0]['meta']['pid1'], batch[0]['meta']['pid2']))])
        elif args.dataset_format == 'nway':
            pred_li.extend([(pid, logits[i].detach().cpu().numpy(), label[i].item())
                            for i, pid in enumerate(batch[0]['meta']['pid'])])
        else:
            raise NotImplementedError

    return pred_li, loss_li


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train analogy model')
    parser.add_argument('--dataset_dir', type=str, required=True, help='dataset dir')
    parser.add_argument('--dataset_format', type=str, choices=['pointwise', 'nway', 'pointwisemerge', 'bow'],
                        default='pointwise', help='dataset format')
    parser.add_argument('--subgraph_file', type=str, required=True, help='entity subgraph file')
    parser.add_argument('--subprop_file', type=str, required=True, help='subprop file')
    parser.add_argument('--emb_file', type=str, default=None, help='embedding file')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--filter_emb', type=str, default=None, help='path to save filtered embedding')
    parser.add_argument('--prep_data', action='store_true', help='preprocess the dataset')
    parser.add_argument('--preped', action='store_true', help='whether to use preprocessed dataset')
    parser.add_argument('--patience', type=int, default=0, help='number of epoch before running out of patience')
    parser.add_argument('--save', type=str, default=None, help='path to save the model')
    parser.add_argument('--load', type=str, default=None, help='path to load the model from')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers used to load data')
    parser.add_argument('--show_progress', action='store_true', help='whether to show training progress')

    parser.add_argument('--method', type=str, default='emb', help='which model to use')
    parser.add_argument('--match', type=str, default='concat', help='how to match two embeddings')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--edge_type', type=str, default='only_property', help='how to form the graph')

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    random.seed(2019)
    np.random.seed(2019)
    torch.manual_seed(2019)

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # configs
    method = args.method
    lr = {'emb': 0.005, 'ggnn': 0.0001, 'cosine': 0.005, 'bow': 0.0001}[method]
    if args.lr is not None:
        lr = args.lr
    debug = False
    keep_one_per_prop = {'emb': True, 'ggnn': False, 'cosine': True, 'bow': False}[method]
    use_cache = False
    use_pseudo_property = False
    show_progress = args.show_progress
    edge_type = args.edge_type
    num_workers = args.num_workers
    batch_size = args.batch_size
    num_class_dict = {'nway': 16, 'pointwise': 1, 'pointwisemerge': 1, 'bow': 1}
    num_graph_dict = {'nway': 1, 'pointwise': 2, 'pointwisemerge': 1, 'bow': 2}
    layer_timesteps = [3, 3]
    Dataset = eval(args.dataset_format.capitalize() + 'Dataset')
    get_dataset_filepath = lambda split, preped=False: os.path.join(
        args.dataset_dir, split + '.' + args.dataset_format + ('.{}.prep'.format(edge_type) if preped else ''))
    if args.dataset_format == 'pointwise':
        accuracy = accuracy_pointwise
    elif args.dataset_format == 'nway':
        accuracy = accuracy_nway
    else:
        accuracy = accuracy_pointwise

    # load properties
    subprops = read_subprop_file(args.subprop_file)
    pid2plabel = dict(p[0] for p in subprops)

    # load subgraph
    if args.preped:
        subgraph_dict = None
    else:
        subgraph_dict = read_subgraph_file(args.subgraph_file, only_root=method in {'emb', 'cosine'})

    # filter emb and get all the properties used by dry run datasets
    if args.filter_emb:
        if method != 'ggnn':
            raise Exception('use ggnn to filter emb')
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
        filer_embedding(args.emb_file, args.filter_emb, set(all_ids.keys()))
        exit(1)

    # load embedding
    emb_id2ind, emb = read_embeddings_from_text_file(
        args.emb_file, debug=debug, emb_size=200, use_padding=True) \
        if args.emb_file else (None, None)
    #properties_as_relations = {'P31', 'P21', 'P527', 'P17'}  # for debug
    properties_as_relations = load_tsv_as_dict(os.path.join(args.dataset_dir, 'pid2ind.tsv'), valuefunc=int)
    # add agg property and node
    properties_as_relations[AGG_PROP] = len(properties_as_relations)
    emb_id2ind[AGG_NODE] = len(emb_id2ind)
    emb = np.concatenate([emb, np.zeros((1, emb.shape[1]), dtype=np.float32)], axis=0)
    # set num_edge_types
    if edge_type == 'one':
        num_edge_types = 1
    elif edge_type == 'only_property':
        num_edge_types = 2  # head and tail
    elif edge_type == 'property':
        num_edge_types = len(properties_as_relations)
    elif edge_type == 'bow':
        num_edge_types = 2  # head and tail
    else:
        raise NotImplementedError
    if args.dataset_format == 'pointwisemerge':
        num_edge_types += 1

    # load data
    dataset_params = {
        'subgraph_dict': subgraph_dict,
        'properties_as_relations': properties_as_relations,
        'emb_id2ind': emb_id2ind,
        'padding': PADDING,
        'edge_type': edge_type,
        'keep_one_per_prop': keep_one_per_prop,
        'use_cache': use_cache,
        'use_pseudo_property': use_pseudo_property,
        'use_top': True,  # TODO: set using external params
    }
    get_dataloader = lambda ds, shuffle, sampler=None: \
        DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                   num_workers=num_workers, collate_fn=ds.collate_fn)

    if issubclass(Dataset, PointwiseDataset):
        # TODO: set neg_ratio
        train_data = Dataset(get_dataset_filepath('train', args.preped), **dataset_params, neg_ratio=None)
    else:
        train_data = Dataset(get_dataset_filepath('train', args.preped), **dataset_params)
    if method == 'emb' or method == 'cosine':
        train_dataloader = get_dataloader(train_data, True)
    else:
        train_dataloader = get_dataloader(
            train_data, False, RandomSampler(train_data, replacement=True, num_samples=50000))
    dev_data = Dataset(get_dataset_filepath('dev', args.preped), **dataset_params)
    dev_dataloader = get_dataloader(dev_data, False)
    test_data = Dataset(get_dataset_filepath('test', args.preped), **dataset_params)
    test_dataloader = get_dataloader(test_data, False)

    if args.prep_data:
        for split in ['train', 'dev', 'test']:
            prep_filepath = get_dataset_filepath(split, True)
            ds = eval(split + '_data')
            print('prep to {}'.format(prep_filepath))
            with open(prep_filepath, 'w') as fout:
                for i in tqdm(range(len(ds))):
                    fout.write(ds.item_to_str(ds[i]) + '\n')
        exit(1)

    # config model, optimizer and evaluation
    if method == 'bow':
        model = Model(num_class=num_class_dict[args.dataset_format],  # number of classes for prediction
                      num_graph=num_graph_dict[args.dataset_format],  # number of graphs in data dict
                      emb_size=64,
                      vocab_size=len(emb_id2ind),
                      padding_ind=0,
                      emb=None,
                      hidden_size=64,
                      method='bow',
                      match=args.match)
    else:
        model = Model(num_class=num_class_dict[args.dataset_format],
                      num_graph=num_graph_dict[args.dataset_format],
                      emb=emb,
                      padding_ind=0,  # the first position is padding embedding
                      hidden_size=64,
                      method=method,
                      num_edge_types=num_edge_types,
                      layer_timesteps=layer_timesteps,
                      match=args.match)
    if args.load:
        print('load model from {}'.format(args.load))
        model.load_state_dict(torch.load(args.load))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    '''
    train_prop_set = set(read_prop_file(os.path.join(args.dataset_dir, 'train.prop')))
    train_metirc = AnalogyEval(args.subprop_file, method='parent', metric='auc_map',
                               reduction='property', prop_set=train_prop_set)
    dev_prop_set = set(read_prop_file(os.path.join(args.dataset_dir, 'dev.prop')))
    dev_metric = AnalogyEval(args.subprop_file, method='parent', metric='auc_map',
                             reduction='property', prop_set=dev_prop_set, debug=False)
    '''

    # train and test
    pat, best_dev_metric = 0, 0
    for epoch in range(300):
        # init performance
        if epoch == 0:
            dev_pred, dev_loss = one_epoch(args, 'dev', dev_dataloader, optimizer,
                                           device=device, show_progress=show_progress)
            print('init')
            #print(np.mean(dev_loss), dev_metric.eval(dev_pred))
            print(np.mean(dev_loss), accuracy(dev_pred, agg='max')[0])

        # train, dev, test
        train_pred, train_loss = one_epoch(args, 'train', train_dataloader, optimizer,
                                           device=device, show_progress=show_progress)
        dev_pred, dev_loss = one_epoch(args, 'dev', dev_dataloader, optimizer,
                                       device=device, show_progress=show_progress)
        test_pred, test_loss = one_epoch(args, 'test', test_dataloader, optimizer,
                                         device=device, show_progress=show_progress)

        # evaluate
        train_metric, _ = accuracy(train_pred, agg='max')
        dev_metric, _ = accuracy(dev_pred, agg='max')
        test_metric, test_ranks = accuracy(test_pred, agg='max')
        print('epoch {:4d}\ttr_loss: {:>.3f}\tdev_loss: {:>.3f}\tte_loss: {:>.3f}'.format(
            epoch + 1, np.mean(train_loss), np.mean(dev_loss), np.mean(test_loss)), end='')
        print('\t\ttr_acc: {:>.3f}\tdev_acc: {:>.3f}\ttest_acc: {:>.3f}'.format(
            train_metric, dev_metric, test_metric))
        rank_to_csv(test_ranks, 'ranks.csv', key2name=pid2plabel)
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
                if args.save:
                    # save epoch with best dev metric
                    torch.save(model.state_dict(), args.save)
