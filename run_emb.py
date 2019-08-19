#!/usr/bin/env python


from typing import List, Tuple, Dict
import argparse
import random
import os
import numpy as np
import torch
from wikiutil.util import read_embeddings_from_text_file
from wikiutil.metric import rank_to_csv
from wikiutil.property import read_subprop_file, get_pid2plabel
from analogy.emb import compute_overlap


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train analogy model')
    parser.add_argument('--dataset_dir', type=str, required=True, help='dataset dir')
    parser.add_argument('--split', type=str, required=True, help='splits separated by :')
    parser.add_argument('--subprop_file', type=str, required=True, help='subprop file')
    parser.add_argument('--emb_file', type=str, default=None, help='embedding file')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--method', type=str, default='avg', help='which model to use')
    parser.add_argument('--filter_num_poccs', type=int, default=None, help='properties larger than this are kept')
    parser.add_argument('--skip_split', action='store_true')
    parser.add_argument('--only_prop_emb', action='store_true')
    parser.add_argument('--top', type=int, default=100)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--use_norm', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    emb_id2ind, emb = read_embeddings_from_text_file(
        args.emb_file, debug=False, emb_size=200, use_padding=True)
    subprops = read_subprop_file(args.subprop_file)
    pid2plabel = get_pid2plabel(subprops)

    ranks = compute_overlap(
        args.dataset_dir,
        args.split.split(':'),
        'poccs.pickle',
        args.subprop_file,
        emb,
        emb_id2ind,
        top=args.top,
        method=args.method,
        only_prop_emb=args.only_prop_emb,
        detect_cheat=False,
        use_minus=False,
        filter_num_poccs=args.filter_num_poccs,
        filter_pids=None,
        num_workers=args.num_workers,
        skip_split=args.skip_split,
        ori_subprops='data/subprops.txt',
        debug=args.debug,
        sigma=args.sigma,
        use_norm=args.use_norm)

    if args.save:
        if not os.path.exists(args.save):
            os.mkdir(args.save)
        rank_to_csv(ranks, os.path.join(args.save, 'ranks.csv'), key2name=pid2plabel)
