from typing import Dict, List, Tuple
import os
import json
from collections import defaultdict
import numpy as np


class FewRelDataset():
    def __init__(self, datadir: str, context_method: str):
        self.train_path = os.path.join(datadir, 'train.json')
        self.val_path = os.path.join(datadir, 'val.json')

        with open(self.train_path) as fin:
            train_raw = json.load(fin)
        with open(self.val_path) as fin:
            val_raw = json.load(fin)
        self.all_raw = self.merge_fewrel_json(train_raw, val_raw)

        self.context_method = context_method


    def merge_fewrel_json(self, *args):
        merged = defaultdict(list)
        for arg in args:
            for pid, sents in arg.items():
                merged[pid].extend(sents)
        return dict(merged)


    def iter(self):
        num_multi_spans = 0
        num_sents = 0
        for pid, sents in self.all_raw.items():
            num_sents += len(sents)
            for sent in sents:
                tokens = np.array(sent['tokens'])
                hid = sent['h'][1]
                hrange = sent['h'][2]
                tid = sent['t'][1]
                trange = sent['t'][2]
                if len(hrange) > 1 or len(trange) > 1:
                    num_multi_spans += 1
                    continue
                hrange, trange = hrange[0], trange[0]
                yield pid, tokens, hid, tid, hrange, trange
        print('{} out of {} sents have multi-span entities'.format(num_multi_spans, num_sents))


    def build_pid2context(self, in_place=False):
        ''' build a mapping from pid to context, which is a dictionary of words and their counts '''
        pid2context: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
        for pid, sent, hid, tid, hrange, trange in self.iter():
            context_words = self.get_context_words(sent, hrange, trange, self.context_method)
            for w in context_words:
                pid2context[pid][w] += 1
        pid2context = dict(pid2context)
        if not in_place:
            return pid2context
        self.pid2context = pid2context


    def build_entity2sent(self, in_place=False):
        sent2sid: Dict[Tuple, int] = defaultdict(lambda: len(sent2sid))
        entity2sid: Dict[str, Dict[int, List[int]]] = defaultdict(dict)
        for pid, sent, hid, tid, hrange, trange in self.iter():
            sent_key = tuple(sent)
            sid = sent2sid[sent_key]
            entity2sid[hid][sid] = hrange
            entity2sid[tid][sid] = trange
        sid2sent = dict((v, list(k)) for k, v in sent2sid.items())
        entity2sid = dict(entity2sid)
        if not in_place:
            return sid2sent, entity2sid
        self.sid2sent, self.entity2sid = sid2sent, entity2sid


    def get_context_words(self,
                          sent: List[str],
                          e1_pos: List[int],
                          e2_pos: List[int],
                          method: str = 'middle') -> List[str]:
        assert method in {'middle'}
        if method == 'middle':
            context = []
            if e1_pos[-1] < e2_pos[0]:
                context = sent[e1_pos[-1]:e2_pos[0]]
            elif e1_pos[0] > e2_pos[-1]:
                context = sent[e2_pos[-1]:e1_pos[0]]
            return context


    def get_coocc(self, hids: List[str], tids: List[str]):
        pocc2context: Dict[str, int] = defaultdict(lambda: 0)
        for hid, tid in zip(hids, tids):
            if hid not in self.entity2sid or tid not in self.entity2sid:
                continue
            h_sids = set(self.entity2sid[hid].keys())
            t_sids = set(self.entity2sid[tid].keys())
            for sid in h_sids & t_sids:
                sent = self.sid2sent[sid]
                hrange = self.entity2sid[hid][sid]
                trange = self.entity2sid[tid][sid]
                context_words = self.get_context_words(sent, hrange, trange, method=self.context_method)
                for w in context_words:
                    pocc2context[w] += 1
        return dict(pocc2context)
