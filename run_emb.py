#!/usr/bin/env python


from typing import List, Tuple, Dict
import argparse, random
import numpy as np
import torch
from wikiutil.util import read_embeddings_from_text_file
from analogy.emb import compute_overlap


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train analogy model')
    parser.add_argument('--dataset_dir', type=str, required=True, help='dataset dir')
    parser.add_argument('--split', type=str, required=True, help='splits separated by :')
    parser.add_argument('--subprop_file', type=str, required=True, help='subprop file')
    parser.add_argument('--emb_file', type=str, default=None, help='embedding file')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')

    parser.add_argument('--method', type=str, default='avg', help='which model to use')
    parser.add_argument('--filter_num_poccs', type=int, default=None, help='properties larger than this are kept')
    parser.add_argument('--skip_split', action='store_true')
    args = parser.parse_args()

    random.seed(2019)
    np.random.seed(2019)
    torch.manual_seed(2019)

    emb_id2ind, emb = read_embeddings_from_text_file(
        args.emb_file, debug=False, emb_size=200, use_padding=True)

    compute_overlap(args.dataset_dir,
                    args.split.split(':'),
                    'poccs.pickle',
                    args.subprop_file,
                    emb,
                    emb_id2ind,
                    top=100,
                    method=args.method,
                    only_prop_emb=False,
                    detect_cheat=False,
                    use_minus=False,
                    filter_num_poccs=args.filter_num_poccs,
                    filter_pids=None,
                    num_workers=args.num_workers,
                    skip_split=args.skip_split,
                    ori_subprops='data/subprops.txt',
                    sigma=1.0)
