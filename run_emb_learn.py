#!/usr/bin/env python


import argparse
import random
import os
import numpy as np
import torch
from analogy.emb_learn import run_emb_train
from wikiutil.property import read_subprop_file, get_pid2plabel
from wikiutil.metric import rank_to_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train analogy model')
    parser.add_argument('--dataset_dir', type=str, required=True, help='dataset dir')
    parser.add_argument('--subprop_file', type=str, required=True, help='subprop file')
    parser.add_argument('--emb_file', type=str, default=None, help='embedding file')
    parser.add_argument('--emb_size', type=int, default=200)
    parser.add_argument('--word_emb_file', type=str, default=None, help='word embedding file')
    parser.add_argument('--word_emb_size', type=int, default=50)
    parser.add_argument('--use_tbow', type=int, default=0)
    parser.add_argument('--suffix', type=str, default='.tbow')
    parser.add_argument('--word_emb_file2', type=str, default=None, help='word embedding file')
    parser.add_argument('--word_emb_size2', type=int, default=50)
    parser.add_argument('--use_tbow2', type=int, default=0)
    parser.add_argument('--suffix2', type=str, default='.tbow')

    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--save', type=str, default=None)

    parser.add_argument('--optimizer', type=str, default='rmsprop')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--only_one_sample_per_prop', action='store_true')
    parser.add_argument('--filter_labels', action='store_true')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_size = args.emb_size * 2 + \
                 (args.word_emb_size if args.use_tbow else 0) + \
                 (args.word_emb_size2 if args.use_tbow2 else 0)
    subprops = read_subprop_file(args.subprop_file)
    pid2plabel = get_pid2plabel(subprops)

    metrics, test_ranks, dev_ranks, train_ranks = run_emb_train(
        args.dataset_dir,
        args.emb_file,
        args.subprop_file,
        use_label=False,
        filter_labels=args.filter_labels,
        filter_leaves=False,
        only_test_on=None,
        optimizer=args.optimizer,
        epoch=args.epoch,
        batch_size=128,
        use_cuda=True,
        early_stop=args.early_stop,
        num_occs=10,
        num_occs_label=200,
        input_size=input_size,
        hidden_size=128,
        lr=args.lr,
        dropout=0.5,
        only_prop=False,
        use_tbow=args.use_tbow,
        tbow_emb_size=args.word_emb_size,
        word_emb_file=args.word_emb_file,
        suffix=args.suffix,
        use_tbow2=args.use_tbow2,
        tbow_emb_size2=args.word_emb_size2,
        word_emb_file2=args.word_emb_file2,
        suffix2=args.suffix2,
        only_tbow=False,
        renew_word_emb=False,
        output_pred=False,
        use_ancestor=False,
        acc_topk=1,
        use_weight=True,
        only_one_sample_per_prop=args.only_one_sample_per_prop)

    print('final metrics: {}'.format(np.mean(metrics[-50:])))

    if args.save:
        if not os.path.exists(args.save):
            os.mkdir(args.save)
        rank_to_csv(test_ranks, os.path.join(args.save, 'ranks.csv'), key2name=pid2plabel)
