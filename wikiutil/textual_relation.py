from typing import Dict, List, Tuple, Iterable
import os
import json
from collections import defaultdict
import numpy as np
import bz2
from tqdm import tqdm
import spacy
import pickle


class TextualDataset:
    def __init__(self, context_method: str):
        self.context_method = context_method


    @classmethod
    def from_entity2sent(cls, pickle_dir: str, context_method: str):
        print('load from {} ...'.format(pickle_dir))
        with open(os.path.join(pickle_dir, 'sid2sent.pkl'), 'rb') as fin:
            sid2sent = pickle.load(fin)
        with open(os.path.join(pickle_dir, 'entity2sid.pkl'), 'rb') as fin:
            entity2sid = pickle.load(fin)
        inst = cls(context_method=context_method)
        inst.sid2sent = sid2sent
        inst.entity2sid = entity2sid
        return inst


class FewRelDataset(TextualDataset):
    def __init__(self, context_method: str = 'middle', datadir: str = None):
        super(FewRelDataset, self).__init__(context_method=context_method)
        if datadir:
            self.train_path = os.path.join(datadir, 'train.json')
            self.val_path = os.path.join(datadir, 'val.json')

            with open(self.train_path) as fin:
                train_raw = json.load(fin)
            with open(self.val_path) as fin:
                val_raw = json.load(fin)
            self.all_raw = self.merge_fewrel_json(train_raw, val_raw)


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


class WikipediaArticleWithLinks:
    def __init__(self,
                 url: str,
                 id: int,
                 text: str,
                 annotations: List[Tuple[int, int, str, str]] = None  # (from, to, id, span)
                 ):
        self.url = url
        self.title = url.rsplit('/', 1)[1]  # TODO: this is problematic
        self.id = id
        self.text = text
        self.annotations = annotations


    @classmethod
    def from_json(cls, json_obj):
        annotations = [(a['offset'], a['offset'] + len(a['surface_form']), a['uri'], a['surface_form'])
                       for a in json_obj['annotations']]
        return cls(url=json_obj['url'], id=json_obj['id'][0], text=json_obj['text'], annotations=annotations)


    @classmethod
    def from_json_v2(cls, json_obj):
        annotations = [(a['from'], a['to'], a['id'], a['label']) for a in json_obj['annotations']]
        return cls(url=json_obj['url'], id=json_obj['id'], text=json_obj['text'], annotations=annotations)


class WikipediaDataset(TextualDataset):
    def __init__(self, context_method: str = 'middle', data_dir: str = None, use_bz2: bool = True):
        super(WikipediaDataset, self).__init__(context_method=context_method)
        self.data_dir = data_dir
        self.use_bz2 = use_bz2
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])


    def wiki_bz2_iter(self, filepath) -> Iterable[WikipediaArticleWithLinks]:
        with bz2.BZ2File(filepath, 'r') as fin:
            for l in fin:
                wiki_article = WikipediaArticleWithLinks.from_json(json.loads(l))
                yield wiki_article


    def build_entity2sent(self,
                          wikipedia_titles: set,
                          max_num_sent: int = None,
                          in_place: bool = False):
        ''' build entity2sent mapping for a set of wikipedia entities '''
        sid2sent = {}
        entity2sid: Dict[str, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
        for root, dirs, files in tqdm(os.walk(self.data_dir, followlinks=True)):
            for file in files:
                if self.use_bz2 and not file.endswith('.bz2'):
                    continue
                for wiki_article in self.wiki_bz2_iter(os.path.join(root, file)):
                    for ann in wiki_article.annotations:
                        from_ind, to_ind, wikititle, span = ann
                        if wikititle not in wikipedia_titles:
                            continue
                        sid = wiki_article.id
                        sent = wiki_article.text
                        if max_num_sent and len(entity2sid[wikititle]) >= max_num_sent:
                            # skip entities already having a lot of sentences
                            continue
                        sid2sent[sid] = sent
                        entity2sid[wikititle][sid].add((from_ind, to_ind))
        entity2sid: Dict[str, Dict[int, List]] = dict((entity, dict((k, list(v)) for k, v in sents.items()))
                                                      for entity, sents in entity2sid.items())

        print('{} out of {} wikipedia titles appears in the text'.format(
            len(entity2sid), len(wikipedia_titles)))
        not_exit = [wt for wt in wikipedia_titles if wt not in entity2sid]
        print('some of the wikipedia titles that do not exist: {}'.format(not_exit[:10]))

        if not in_place:
            return sid2sent, entity2sid
        self.sid2sent = sid2sent
        self.entity2sid = entity2sid


    def build_entity2sent_for_wikidata(self,
                                       wikidata_ids: set,
                                       wikidata2wikipedia_title: Dict[str, str],
                                       max_num_sent: int = None,
                                       in_place: bool = False):
        ''' build entity2sent mapping for a set of wikidata entities '''
        wikipedia_titles = set(wikidata2wikipedia_title[wid] for wid in wikidata_ids
                               if wid in wikidata2wikipedia_title)
        print('{} out of {} wikidata ids have wikipedia titles'.format(
            len(wikipedia_titles), len(wikidata_ids)))
        not_exist = [wid for wid in wikidata_ids if wid not in wikidata2wikipedia_title]
        print('some of wikidata ids that do not exist: {}'.format(not_exist[:10]))

        sid2sent, entity2sid = self.build_entity2sent(wikipedia_titles,
                                                      max_num_sent=max_num_sent,
                                                      in_place=False)
        wikipedia_title2wikidata = dict((v, k) for k, v in wikidata2wikipedia_title.items())
        entity2sid = dict((wikipedia_title2wikidata[k], v) for k, v in entity2sid.items())

        if not in_place:
            return sid2sent, entity2sid
        self.sid2sent = sid2sent
        self.entity2sid = entity2sid


    def get_context_words(self,
                          sent: str,
                          e1_pos: List[Tuple[int, int]],
                          e2_pos: List[Tuple[int, int]],
                          tokenize: bool = True,
                          dist_thres: int = None,
                          method: str = 'middle') -> List[str]:
        assert method in {'middle'}
        # find the closest e1 and e2
        e1_pos_mid = np.array([(s + e) / 2 for s, e in e1_pos])
        e2_pos_mid = np.array([(s + e) / 2 for s, e in e2_pos])
        dist = np.abs(e1_pos_mid.reshape(-1, 1) - e2_pos_mid.reshape(1, -1))
        pos = np.argmin(dist)
        e1_pos = e1_pos[pos // len(e2_pos_mid)]
        e2_pos = e2_pos[pos % len(e2_pos_mid)]

        # build context
        if method == 'middle':
            if e1_pos[-1] < e2_pos[0]:
                context = sent[e1_pos[-1]:e2_pos[0]]
            elif e1_pos[0] > e2_pos[-1]:
                context = sent[e2_pos[-1]:e1_pos[0]]
            else:  # overlap entities
                return []
            if len(context) >= 1000:  # TODO: a heurist to avoid long distence
                return []
            if tokenize:
                doc = self.nlp(context)
                context = [t.text for t in doc]
                if dist_thres and len(context) > dist_thres:  # two entities are too far from each other
                    return []
            return context


    def get_coocc(self, hids: List[str], tids: List[str], dist_thres: int):
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
                context_words = self.get_context_words(
                    sent, hrange, trange, tokenize=True, dist_thres=dist_thres, method=self.context_method)
                for w in context_words:
                    pocc2context[w] += 1
        return dict(pocc2context)
