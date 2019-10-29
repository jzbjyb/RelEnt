import argparse
import pickle
from random import shuffle
import json
from wikiutil.property import read_prop_occ_file_from_dir
from wikiutil.textual_feature import filter_snippet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snippet', help='snippet data path', required=True)
    parser.add_argument('--property', help='property occurrence dir', required=False)
    parser.add_argument('--eid2name', help='entity id to name file', required=False)
    parser.add_argument('--out', help='output', required=True)
    args = parser.parse_args()

    with open(args.snippet, 'rb') as fin:
        pid2snippet = pickle.load(fin)

    #with open(args.eid2name, 'rb') as fin:
    #    eid2name = pickle.load(fin)

    # filter snippets
    pid2snippet = dict((pid, filter_snippet(dict((w, c) for w, c in wd.items() if c >= 10))) for pid, wd in pid2snippet.items())
    pid2snippet = dict((pid, wd) for pid, wd in pid2snippet.items() if len(wd) > 0)

    result = {}
    for pid, snippets in pid2snippet.items():
        snippets = sorted(snippets.items(), key=lambda x: -x[1])
        #occs = read_prop_occ_file_from_dir(pid, args.property, filter=False, contain_name=False, use_order=False)
        #occs = list(occs)
        #shuffle(occs)
        #occs = [(eid2name[h], eid2name[t]) for h, t in occs if h in eid2name and t in eid2name][:100]
        snippets = snippets[:50]
        result[pid] = {
            'snippet': snippets,
            #'occs': occs
        }

    with open(args.out, 'w') as fout:
        json.dump(result, fout, indent=True)
