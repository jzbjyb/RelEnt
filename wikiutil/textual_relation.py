from typing import Dict, List, Tuple, Iterable
import os
import json
from collections import defaultdict
import numpy as np
import bz2
from tqdm import tqdm
import spacy
from spacy.tokens import Doc
import pickle
from copy import deepcopy
from random import shuffle

ner_nlp = spacy.load('xx_ent_wiki_sm', disable=['parser', 'tagger'])


def get_dep_path(doc: spacy.tokens.Doc, e1_idx: set, e2_idx: set) -> List[str]:
    # build parent -> (child, dep) and child -> (parent, dep) mappings
    p2c: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    c2p: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for c, w in enumerate(doc):
        p = w.head.i
        dep = w.dep_
        p2c[p].append((c, dep))
        c2p[c].append((p, dep + '_'))

    # search starting from each e1 index
    e1_idx = list(e1_idx)
    to_go = [idx for idx in e1_idx]
    history = [[] for idx in e1_idx]
    went = set(to_go)
    shortest = []  # shortest path from e1_idx to e2_idx
    while len(to_go) > 0:
        next_to_go = []
        next_history = []
        for idx, hist in zip(to_go, history):
            # to parent and child
            for direction in [p2c, c2p]:
                if idx not in direction:
                    continue
                for next, dep in direction[idx]:
                    if next not in went:
                        went.add(next)
                        next_to_go.append(next)
                        nhist = deepcopy(hist)
                        if len(nhist) > 0:  # not include starting node
                            nhist.append(doc[idx].text)
                        nhist.append(dep)
                        next_history.append(nhist)
                        if next in e2_idx:
                            shortest = nhist
                            break
                if shortest:
                    break
            if shortest:
                break
        to_go = next_to_go
        history = next_history
        if shortest:
            break

    return shortest


class CharTokenizer(object):
    def __init__(self, vocab, char: str = ' '):
        self.vocab = vocab
        self.char = char


    def __call__(self, text):
        words = text.split(self.char)
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        spaces[-1] = False
        return Doc(self.vocab, words=words, spaces=spaces)


class TextualDataset:
    @classmethod
    def from_entity2sent(cls, pickle_dir: str):
        print('load from {} ...'.format(pickle_dir))
        with open(os.path.join(pickle_dir, 'sid2sent.pkl'), 'rb') as fin:
            sid2sent = pickle.load(fin)
        with open(os.path.join(pickle_dir, 'entity2sid.pkl'), 'rb') as fin:
            entity2sid = pickle.load(fin)
        inst = cls()
        inst.sid2sent = sid2sent
        inst.entity2sid = entity2sid
        return inst


class FewRelDataset(TextualDataset):
    def __init__(self, datadir: str = None):
        super(FewRelDataset, self).__init__()
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.tokenizer = CharTokenizer(self.nlp.vocab, char='\t')
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


    def iter(self, by_pid=False):
        num_multi_spans = 0
        num_sents = 0
        for pid, sents in self.all_raw.items():
            num_sents += len(sents)
            sent_li, hid_li, tid_li, hrange_li, trange_li = [], [], [], [], []
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
                if by_pid:
                    sent_li.append(tokens)
                    hid_li.append(hid)
                    tid_li.append(tid)
                    hrange_li.append(hrange)
                    trange_li.append(trange)
                else:
                    yield pid, tokens, hid, tid, hrange, trange
            if by_pid:
                yield pid, sent_li, hid_li, tid_li, hrange_li, trange_li

        print('{} out of {} sents have multi-span entities'.format(num_multi_spans, num_sents))


    def build_pid2context(self,
                          dist_thres: int,
                          context_method: str,
                          in_place: bool = False,
                          max_num_sent: int = None):
        ''' build a mapping from pid to context, which is a dictionary of words and their counts '''
        pid2context: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
        for pid, sent_li, hid_li, tid_li, hrange_li, trange_li in tqdm(self.iter(by_pid=True)):
            if max_num_sent:
                sent_li = sent_li[:max_num_sent]
                hrange_li = hrange_li[:max_num_sent]
                trange_li = trange_li[:max_num_sent]
            context_words_li = self.get_context_words(
                sent_li, hrange_li, trange_li, method=context_method, dist_thres=dist_thres)
            for context_words in context_words_li:
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
                          sent_li: List[List[str]],
                          e1_pos_li: List[List[int]],
                          e2_pos_li: List[List[int]],
                          method: str = 'middle',
                          dist_thres: int = None) -> List[List[str]]:
        assert method in {'middle', 'dep'}
        context_li = []

        if method == 'middle':
            for sent, e1_pos, e2_pos in zip(sent_li, e1_pos_li, e2_pos_li):
                context = []
                if e1_pos[-1] < e2_pos[0]:
                    context = sent[e1_pos[-1]:e2_pos[0]]
                elif e1_pos[0] > e2_pos[-1]:
                    context = sent[e2_pos[-1]:e1_pos[0]]
                if dist_thres and len(context) > dist_thres:
                    context = []
                context_li.append(context)

        elif method == 'dep':
            sent_li = ['\t'.join(sent) for sent in sent_li]
            doc_li = self.nlp.pipe(sent_li)
            for doc, e1_pos, e2_pos in zip(doc_li, e1_pos_li, e2_pos_li):
                dep_path = get_dep_path(doc, set(e1_pos), set(e2_pos))
                if len(dep_path) <= dist_thres:
                    context_li.append(dep_path)

        return context_li


    def get_coocc(self, hids: List[str], tids: List[str], context_method: str, dist_thres: int):
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
                    sent, hrange, trange, method=context_method, dist_thres=dist_thres)
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
    def __init__(self, data_dir: str = None, use_bz2: bool = True):
        super(WikipediaDataset, self).__init__()
        self.data_dir = data_dir
        self.use_bz2 = use_bz2
        #self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
        self.nlp = spacy.load('en_core_web_sm')


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
                          sent_li: List[str],
                          e1_pos_li: List[List[Tuple[int, int]]],
                          e2_pos_li: List[List[Tuple[int, int]]],
                          tokenize: bool = True,
                          dist_thres: int = None,
                          method: str = 'middle') -> List[List[str]]:
        assert method in {'middle', 'dep'}

        # --- collect pieces for spacy ---
        to_spacy_li = []
        if method == 'middle':
            pass
        elif method == 'dep':
            boundary_li = []

        for sent, e1_poss, e2_poss in zip(sent_li, e1_pos_li, e2_pos_li):
            # find the closest e1 and e2
            e1_pos_mid = np.array([(s + e) / 2 for s, e in e1_poss])
            e2_pos_mid = np.array([(s + e) / 2 for s, e in e2_poss])
            dist = np.abs(e1_pos_mid.reshape(-1, 1) - e2_pos_mid.reshape(1, -1))
            for pos in np.argsort(dist.reshape(-1)):
                e1_pos = e1_poss[pos // len(e2_pos_mid)]
                e2_pos = e2_poss[pos % len(e2_pos_mid)]

                # build context
                if method == 'middle':
                    if e1_pos[-1] < e2_pos[0]:
                        context = sent[e1_pos[-1]:e2_pos[0]]
                    elif e1_pos[0] > e2_pos[-1]:
                        context = sent[e2_pos[-1]:e1_pos[0]]
                    else:  # overlap entities
                        continue
                    if len(context) >= 1000:  # TODO: a heurist to avoid long distence
                        continue
                    to_spacy_li.append(context)
                elif method == 'dep':
                    st1, ed1 = e1_pos[0], e1_pos[-1]
                    st2, ed2 = e2_pos[0], e2_pos[-1]
                    if ed1 <= st2:
                        context = sent[ed1:st2]
                    elif ed2 <= st1:
                        context = sent[ed2:st1]
                    else:  # overlap entities
                        continue
                    if len(context) >= 100:  # TODO: a heurist to avoid long distence
                        continue

                    # locate the paragraph using newline
                    para_st, para_ed = 0, len(sent)  # default is the first and last token of the whole document
                    for i in range(min(st1, st2) - 1, -1, -1):  # find the start location of the paragraph
                        if sent[i] == '\n':
                            para_st = i + 1
                            break
                    for i in range(max(ed1, ed2), len(sent)):  # find the end location of the paragraph
                        if sent[i] == '\n':
                            para_ed = i
                            break
                    para = sent[para_st:para_ed]
                    st1, ed1, st2, ed2 = [idx - para_st for idx in [st1, ed1, st2, ed2]]

                    to_spacy_li.append(para)
                    boundary_li.append((st1, ed1, st2, ed2))

        # do spacy
        context_li = []
        if method == 'middle':
            if tokenize:
                doc_li = self.nlp.pipe(to_spacy_li)
                for doc in doc_li:
                    context = [t.text for t in doc]
                    if dist_thres and len(context) <= dist_thres:
                        context_li.append(context)
            else:
                doc_li = to_spacy_li
                for doc in doc_li:
                    context = doc.strip().split()
                    if dist_thres and len(context) <= dist_thres:
                        context_li.append(context)
        elif method == 'dep':
            doc_li = self.nlp.pipe(to_spacy_li)
            for doc, boundary in zip(doc_li, boundary_li):
                st1, ed1, st2, ed2 = boundary
                # identify corresponding words in the paragraph
                e1_idx, e2_idx = set(), set()
                for i, w in enumerate(doc):
                    if w.idx >= st1 and w.idx + len(w.text) <= ed1:
                        e1_idx.add(i)
                    elif w.idx >= st2 and w.idx + len(w.text) <= ed2:
                        e2_idx.add(i)
                dep_path = get_dep_path(doc, e1_idx, e2_idx)
                if len(dep_path) <= dist_thres:
                    context_li.append(dep_path)

        return context_li


    def get_coocc(self,
                  hids: List[str],
                  tids: List[str],
                  context_method: str,
                  dist_thres: int,
                  max_num_sent: int = None,
                  entityname2id: Dict[str, List[str]] = None):

        # get all sentence id
        sids = set()
        for hid, tid in zip(hids, tids):
            if hid not in self.entity2sid or tid not in self.entity2sid:
                continue
            h_sids = set(self.entity2sid[hid].keys())
            t_sids = set(self.entity2sid[tid].keys())
            join = h_sids & t_sids
            any = (h_sids | t_sids) - (h_sids & t_sids)
            join = list(join)
            any = list(any)
            shuffle(join)
            shuffle(any)
            if max_num_sent:
                sids.update((join + any)[:max_num_sent])
            else:
                sids.update(join + any)

        # do ner to get more mentions
        if entityname2id is not None:
            extended_entity2sid = self.extend_entity2sent(entityname2id, sids)
            print('extend {} from {} sentences'.format(len(extended_entity2sid), len(sids)))
        else:
            extended_entity2sid = defaultdict(lambda: defaultdict(set))

        # add original entity2sid
        for hid, tid in zip(hids, tids):
            for id_ in [hid, tid]:
                if id_ in self.entity2sid:
                    for sid in self.entity2sid[id_]:
                        extended_entity2sid[id_][sid].update(self.entity2sid[id_][sid])

        extended_entity2sid: Dict[str, Dict[int, List]] = \
            dict((entity, dict((k, list(v)) for k, v in sents.items()))
                 for entity, sents in extended_entity2sid.items())

        # collect sentences
        sent_li, hrange_li, trange_li = [], [], []
        for hid, tid in zip(hids, tids):
            if hid not in extended_entity2sid or tid not in extended_entity2sid:
                continue
            h_sids = set(extended_entity2sid[hid].keys())
            t_sids = set(extended_entity2sid[tid].keys())
            for i, sid in enumerate(h_sids & t_sids):
                if sid not in sids:
                    continue
                sent = self.sid2sent[sid]
                hrange = extended_entity2sid[hid][sid]
                trange = extended_entity2sid[tid][sid]
                sent_li.append(sent)
                hrange_li.append(hrange)
                trange_li.append(trange)

        # find context
        pocc2context: Dict[str, int] = defaultdict(lambda: 0)
        for context_words in self.get_context_words(
                sent_li, hrange_li, trange_li, tokenize=True, dist_thres=dist_thres, method=context_method):
            for w in context_words:
                pocc2context[w] += 1

        return dict(pocc2context)


    def extend_entity2sent(self,
                           entityname2id: Dict[str, List[str]],
                           sids: set = None,
                           show_progress: bool = False):
        extended_entity2sid: Dict[str, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
        if sids is None:
            sids = self.sid2sent.keys()
        sent_li = []
        for sid in sids:
            if sid not in self.sid2sent:
                continue
            sent_li.append(self.sid2sent[sid])
        for doc in tqdm(ner_nlp.pipe(sent_li), disable=not show_progress):
            for ent in doc.ents:
                ent_str = ent.text
                ent_st = ent.start_char
                ent_ed = ent.end_char
                wiki_entities = match_mention_entity(ent_str, entityname2id)
                for we in wiki_entities:
                    extended_entity2sid[we][sid].add((ent_st, ent_ed))
        return extended_entity2sid


def entity_idname_conversion(entityid2name: Dict[str, List[str]]):
    result: Dict[str, List[str]] = defaultdict(list)
    for k, names in entityid2name.items():
        for name in names:  # TODO: lower?
            result[name].append(k)
    return result


def match_mention_entity(mention: str, entityname2id: Dict[str, List[str]]) -> List[str]:
    mention = mention.lower()
    if mention in entityname2id:
        return entityname2id[mention]
    return []
