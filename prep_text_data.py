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
from wikiutil.textual_feature import dep_prep, middle_prep
from analogy.emb import compute_overlap


if __name__ == '__main__':
    parser = argparse.ArgumentParser('prep textual data')
    parser.add_argument('--data_dir', type=str, required=True, help='dataset dir')
    parser.add_argument('--subprop_file', type=str, required=True, help='subprop file')
    parser.add_argument('--emb_file', type=str, default=None, help='embedding file')

    parser.add_argument('--dep_dir', type=str, default=None)
    parser.add_argument('--pid2snippet_file', type=str, default=None)

    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--suffix', type=str, default=None)

    args = parser.parse_args()

    if args.dep_dir:
        dep_prep(args.dep_dir, args.subprop_file, data_dir=args.data_dir,
                 output=args.out, suffix=args.suffix, emb_file=args.emb_file)
    elif args.pid2snippet_file:
        middle_prep(args.pid2snippet_file, args.subprop_file, data_dir=args.data_dir, entityid2name_file=None,
                    output=args.out, suffix=args.suffix, emb_file=args.emb_file)
