#!/usr/bin/env python


import argparse
import random
import os
import numpy as np
import torch
from analogy.emb_learn import run_emb_train
from wikiutil.property import read_subprop_file, get_pid2plabel
from wikiutil.metric import rank_to_csv
from wikiutil.util import load_tsv_as_dict


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

    parser.add_argument('--sent_emb_file', type=str, default=None, help='sent embedding file')
    parser.add_argument('--sent_emb_size', type=int, default=50)
    parser.add_argument('--use_sent', type=int, default=0)
    parser.add_argument('--sent_suffix', type=str, default='.tbow')

    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--optimizer', type=str, default='rmsprop')
    parser.add_argument('--num_occs', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0)
    parser.add_argument('--lr_decay', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--sent_hidden_size', type=int, default=16)
    parser.add_argument('--only_one_sample_per_prop', action='store_true')
    parser.add_argument('--filter_labels', action='store_true')
    parser.add_argument('--use_label', action='store_true')
    parser.add_argument('--use_gnn', type=str, default=None)
    parser.add_argument('--sent_emb_method', type=str, default='cnn_mean')
    parser.add_argument('--only_text', action='store_true')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_size = args.emb_size * 2 + \
                 (args.word_emb_size if args.use_tbow else 0) + \
                 (args.word_emb_size2 if args.use_tbow2 else 0)
    subprops = read_subprop_file(args.subprop_file)
    pid2plabel = get_pid2plabel(subprops)
    label2ind = load_tsv_as_dict(os.path.join(args.dataset_dir, 'label2ind.txt'), valuefunc=str)

    metrics, test_ranks, dev_ranks, train_ranks = run_emb_train(
        args.dataset_dir,
        args.emb_file,
        args.subprop_file,
        use_label=args.use_label,
        filter_labels=args.filter_labels,
        filter_leaves=False,
        only_test_on=None,
        optimizer=args.optimizer,
        epoch=args.epoch,
        batch_size=args.batch_size,
        use_cuda=not args.no_cuda,
        early_stop=args.early_stop,
        num_occs=args.num_occs,
        num_occs_label=args.num_occs,
        input_size=input_size,
        hidden_size=args.hidden_size,
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
        use_sent=args.use_sent,
        sent_emb_size=args.sent_emb_size,
        sent_emb_file=args.sent_emb_file,
        sent_suffix=args.sent_suffix,
        only_tbow=args.only_text,
        only_sent=args.only_text,
        renew_word_emb=False,
        output_pred=False,
        use_ancestor=False,
        acc_topk=1,
        use_weight=True,
        only_one_sample_per_prop=args.only_one_sample_per_prop,
        use_gnn=args.use_gnn,
        sent_emb_method=args.sent_emb_method,
        sent_hidden_size=args.sent_hidden_size,
        lr_decay=args.lr_decay)

    print('final metrics: {}'.format(np.mean(metrics[-args.early_stop:])))
    print('last metrics: {}'.format(metrics[-1]))

    if args.save:
        if not os.path.exists(os.path.dirname(args.save)):
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
        rank_to_csv(test_ranks, args.save, key2name=pid2plabel,
                    simple_save=True, label2ind=label2ind)
