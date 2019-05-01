#!/usr/bin/env python


import argparse, json
from wikiutil.property import get_sub_properties


def subprop(args):
    all_props = []
    with open(args.inp) as fin:
        props = json.load(fin)
        for prop in props:
            pid, plabel = prop['id'], prop['label']
            all_props.append((pid, plabel))
    all_props = sorted(all_props, key=lambda x: int(x[0][1:]))
    for pid, plabel in all_props:
        subs = get_sub_properties(pid)
        print(pid, plabel, subs)
        input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('process wikidata property')
    parser.add_argument('--task', type=str, choices=['subprop'], required=True)
    parser.add_argument('--inp', type=str, required=True)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    if args.task == 'subprop':
        subprop(args)