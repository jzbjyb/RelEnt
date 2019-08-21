from typing import Dict, List
import os
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
import re
import string
import spacy
import pickle
import numpy as np
from .property import read_subprop_file, get_pid2plabel
from .util import read_embeddings_from_text_file
from .data import read_nway_file


def get_checkword():
    nlp = spacy.load('en_core_web_sm')
    stopwords = nlp.Defaults.stop_words
    word_pattern = re.compile('^[A-Za-z0-9]+$')
    def check_words(word):
        word = word.lower()
        if word in stopwords:
            return None
        if not word_pattern.match(word):
            return None
        return word
    #return check_words
    def filter_bow(words: Dict[str, int], use_direction=False):
        new_words: Dict[str, int] = {}
        if not use_direction:
            for w, c in words.items():
                w = check_words(w)
                if w is not None:
                    new_words[w] = c
        else:
            for (w, dire), c in words.items():
                w = check_words(w)
                if w is not None:
                    new_words[(w, dire)] = c
        return new_words
    return filter_bow
filter_bow = get_checkword()


def get_snippet_filter():
    nlp = spacy.load('en_core_web_sm')
    stopwords = nlp.Defaults.stop_words
    word_pattern = re.compile('^[A-Za-z0-9]+$')
    punct = set(list(string.punctuation))
    def check_snippet(snippet, direction):
        snippet = snippet.lower()
        toks = snippet.split()
        if snippet in punct:  # remove punct
            return None
        if snippet[0] in punct:  # start with punct
            return None
        if np.all([w in stopwords for w in toks]):  # all tokens are stopwords
            return None
        if len(toks) > 5:  # too long
            return None
        return snippet, direction
    def split_snippet(snippet, direction):
        for w in snippet.split():
            if check_words(w):
                yield w, direction
    def filter_snippet(snippets: Dict[str, int]):
        new_snippets: Dict[str, int] = defaultdict(lambda: 0)
        for snippet, c in snippets.items():
            snippet = check_snippet(*snippet)
            if snippet is not None:
                new_snippets[snippet] += c
        return new_snippets
    return filter_snippet
filter_snippet = get_snippet_filter()


def property_level_todata(pid2context,
                          data_dir,
                          subprop_file,
                          emb_file=None,
                          min_count=1,
                          split_sent=False,
                          extend_vocab=False,
                          suffix='.sent'):
    if split_sent and not extend_vocab:
        raise Exception('when split sent, always extend vocab')

    # remove low-frequency sentences
    pid2context = dict((pid, dict((w, c) for w, c in wd.items() if c >= min_count)) for pid, wd in pid2context.items())
    pid2context = dict((pid, wd) for pid, wd in pid2context.items() if len(wd) > 0)

    # sort sentences
    pid2context = dict((pid, sorted(wd.items(), key=lambda x: -x[1])) for pid, wd in pid2context.items())

    print('#pid with context {}'.format(len(pid2context)))

    if emb_file:
        # load embedding
        emb_id2ind, emb = read_embeddings_from_text_file(
            emb_file, debug=False, emb_size=50, first_line=False, use_padding=True,
            split_char=' ', use_cache=True)
        if '<UNK>' in emb_id2ind:
            raise Exception('found <UNK>')
        emb_id2ind['<UNK>'] = len(emb_id2ind)
    else:
        emb_id2ind = defaultdict(lambda: len(emb_id2ind))
        _ = emb_id2ind['<PAD>']

    # save to files
    train_samples = read_nway_file(os.path.join(data_dir, 'train.nway'))
    dev_samples = read_nway_file(os.path.join(data_dir, 'dev.nway'))
    test_samples = read_nway_file(os.path.join(data_dir, 'test.nway'))
    label_samples = read_nway_file(os.path.join(data_dir, 'label2occs.nway'))

    subprops = read_subprop_file(subprop_file)
    pid2plabel = get_pid2plabel(subprops)

    # output pids in test set that have context
    print('pids in test set that have context')
    print(list(map(lambda x: pid2plabel[x], set(pid2context.keys()) & set(s[0][0] for s in test_samples))))

    found_words = set()
    all_words = set()
    for split in ['train', 'label', 'dev', 'test']:
        samples = eval(split + '_samples')
        bow_file = os.path.join(data_dir, split + suffix)
        with open(bow_file, 'w') as fout:
            for i, ((pid, poccs), plabel) in tqdm(enumerate(samples)):
                if pid in pid2context:
                    context = pid2context[pid]
                else:
                    context = []

                first = True
                unk_count = 0

                if not split_sent:
                    for w, c in context:
                        all_words.add(w)
                        # add to vocab
                        if split in {'train', 'label'} and emb_file is None:
                            _ = emb_id2ind[w]
                        if split in {'train', 'label'} and extend_vocab and w not in emb_id2ind:
                            emb_id2ind[w] = len(emb_id2ind)
                        # convert word to ind
                        if w in emb_id2ind:
                            found_words.add(w)
                            if not first:
                                fout.write('\t')
                            fout.write('{} {}'.format(emb_id2ind[w], c))
                            first = False
                        else:
                            unk_count += c
                else:
                    for sent, c in context:
                        toks: List[int] = []
                        for w in sent.split(' '):
                            all_words.add(w)
                            # add to vocab
                            if split in {'train', 'label'} and emb_file is None:
                                _ = emb_id2ind[w]
                            if split in {'train', 'label'} and emb_file and w not in emb_id2ind:
                                emb_id2ind[w] = len(emb_id2ind)
                            # convert word to ind
                            if w in emb_id2ind:
                                found_words.add(w)
                                toks.append(emb_id2ind[w])
                            else:
                                unk_count += c
                                toks.append(emb_id2ind['<UNK>'])
                        if len(toks) > 0:
                            if not first:
                                fout.write('\t')
                            fout.write(' '.join(map(str, toks)))
                            fout.write('\t{}'.format(c))
                            first = False
                fout.write('\n')

    print('found {} out of {}'.format(len(found_words), len(all_words)))
    if emb_file is None:
        with open(os.path.join(data_dir, '{}.vocab'.format(suffix.lstrip('.'))), 'w') as fout:
            for w, i in emb_id2ind.items():
                fout.write('{}\t{}\n'.format(w, i))
    elif extend_vocab:
        num_new_words = len(emb_id2ind) - len(emb)
        emb_dim = emb.shape[1]
        print('#new words {}, #new vocab size {}'.format(num_new_words, len(emb_id2ind)))
        emb_ind2id = dict((v, k) for k, v in emb_id2ind.items())
        os.makedirs(os.path.join(data_dir, 'emb'), exist_ok=True)
        with open(os.path.join(data_dir, 'emb', '{}.txt'.format(suffix.lstrip('.'))), 'w') as fout:
            for i in range(len(emb_ind2id)):
                if i < len(emb):
                    e = emb[i]
                else:
                    e = np.random.uniform(-0.1, 0.1, emb_dim)
                fout.write('{} {}\n'.format(emb_ind2id[i], ' '.join(map(lambda x: '{:.2f}'.format(x), e))))


def dep_prep(dep_dir, subprop_file, data_dir=None, output=False, suffix=None, emb_file=None):
    if suffix and not suffix.startswith('.'):
        raise Exception
    assert output in {False, 'all', 'bow', 'sent'}

    subprops = read_subprop_file(subprop_file)
    pid2plabel = get_pid2plabel(subprops)

    pid2dep: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    for root, dirs, files in tqdm(os.walk(dep_dir)):
        for file in files:
            if root.find('_dep') == -1:
                continue
            if not file.endswith('.txt'):
                continue
            pid = file.rsplit('.', 1)[0]
            with open(os.path.join(root, file), 'r') as fin:
                for l in fin:
                    l = l.rstrip('\n')
                    hid, tid, dep = l.split('\t')
                    pid2dep[pid][dep] += 1
    pid2context: Dict[str, Dict[str, int]] = {}
    for pid, dep in pid2dep.items():
        # filter dep
        dep = dict((d.lower(), c) for d, c in dep.items() if
                   c >= 10 and len(d) > 0 and len(d.split(' ')) > 1)  # count, non-empty, length
        # generate BOW
        if output == 'bow':
            wd_word: Dict[str, int] = defaultdict(lambda: 0)
            wd_dep: Dict[str, int] = defaultdict(lambda: 0)
            new_dep: Dict[str, int] = defaultdict(lambda: 0)
            for raw_d, c in dep.items():
                d = raw_d.split(' ')
                for i, w in enumerate(d):
                    if i % 2 == 1:
                        wd_word[w] += 1
                    else:
                        wd_dep[w] += 1
            wd_word = filter_bow(wd_word)
            for w, c in wd_word.items():
                new_dep[w] += c
            for w, c in wd_dep.items():
                new_dep[w] += c
            dep = new_dep
        if len(dep) > 0:
            pid2context[pid] = dep

    print('#property has dep {}'.format(len(pid2context)))
    for pid, dep in sorted(pid2context.items(), key=itemgetter(0)):
        print(pid2plabel[pid], len(dep), sorted(dep.items(), key=lambda x: -x[1])[0])

    if output == 'all':
        property_level_todata(
            pid2context,
            data_dir,
            subprop_file,
            None,
            min_count=1,
            suffix=suffix)
    elif output == 'bow':
        property_level_todata(
            pid2context,
            data_dir,
            subprop_file,
            emb_file,
            min_count=1,
            extend_vocab=True,
            suffix=suffix)
    elif output == 'sent':
        property_level_todata(
            pid2context,
            data_dir,
            subprop_file,
            emb_file,
            min_count=1,
            extend_vocab=True,
            split_sent=True,
            suffix=suffix
        )


def middle_prep(pid2snippet_file, subprop_file, data_dir=None, entityid2name_file=None, output=False, suffix=None, emb_file=None):
    if suffix and not suffix.startswith('.'):
        raise Exception
    assert output in {False, 'all', 'bow', 'sent'}

    subprops = read_subprop_file(subprop_file)
    if entityid2name_file:
        with open(entityid2name_file, 'rb') as fin:
            entityid2name = pickle.load(fin)
    else:
        entityid2name = None
    pid2plabel = get_pid2plabel(subprops, entityid2name)

    with open(pid2snippet_file, 'rb') as fin:
        pid2snippet = pickle.load(fin)

    # remove low-frequency words
    pid2snippet = dict((pid, dict((w, c) for w, c in wd.items() if c >= 10)) for pid, wd in pid2snippet.items())
    pid2snippet = dict((pid, wd) for pid, wd in pid2snippet.items() if len(wd) > 0)

    # filter
    pid_snippet = []
    pid2context: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    for pid, snippets in sorted(pid2snippet.items(), key=itemgetter(0)):
        # filter snippets
        snippets = filter_snippet(snippets)
        # generate bow
        if output == 'bow':
            word_context: Dict[str, int] = defaultdict(lambda: 0)
            for (snippet, direction), c in snippets.items():
                snippet = snippet.split(' ')
                for i, w in enumerate(snippet):
                    word_context[(w, direction)] += 1
            word_context = filter_bow(word_context, use_direction=True)
            snippets = word_context
        snippets = sorted(snippets.items(), key=lambda x: -x[1])
        if len(snippets) > 0:
            pid_snippet.append((pid, snippets))
            for (w, direction), count in snippets:
                pid2context[pid][w] += count

    print('totally {} pids with snippet'.format(len(pid_snippet)))
    for pid, snippets in pid_snippet:
        print(pid2plabel[pid])
        print(snippets[:5])
        print()

    if output == 'all':
        property_level_todata(
            pid2context,
            data_dir,
            subprop_file,
            None,
            min_count=1,
            suffix=suffix)
    elif output == 'bow':
        property_level_todata(
            pid2context,
            data_dir,
            subprop_file,
            emb_file,
            min_count=1,
            extend_vocab=True,
            suffix=suffix)
    elif output == 'sent':
        property_level_todata(
            pid2context,
            data_dir,
            subprop_file,
            emb_file,
            min_count=1,
            extend_vocab=True,
            split_sent=True,
            suffix=suffix
        )
