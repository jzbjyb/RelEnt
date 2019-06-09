from typing import Dict, Tuple
import functools, os
import pickle
from collections import OrderedDict, Callable, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile
from .constant import PADDING


def load_tsv_as_dict(filepath, keyfunc=lambda x:x, valuefunc=lambda x:x):
    result = {}
    with open(filepath, 'r') as fin:
        for l in fin:
            l = l.strip().split('\t')
            result[keyfunc(l[0])] = valuefunc(l[1])
    return result


def load_tsv_as_list(filepath):
    result = []
    with open(filepath, 'r') as fin:
        for l in fin:
            l = l.strip().split('\t')
            result.append(l)
    return result


def file_filter_by_key(cache_dir):
    def wrapper(func):
        @functools.wraps(func)
        def new_func(rawfile, filterfile=None, filterkeys=None):
            if filterfile is None:
                return func(rawfile, filterkeys=None)
            rf = rawfile.replace('/', '__')
            ff = filterfile.replace('/', '__')
            smallfile = os.path.join(cache_dir, '{}--{}'.format(rf, ff))
            if not os.path.exists(smallfile):
                with open(smallfile, 'w') as fout:
                    for l in func(rawfile, filterkeys=filterkeys)():
                        fout.write('{}\n'.format(l))
            return func(smallfile, filterkeys=None)
        return new_func
    return wrapper


@file_filter_by_key(cache_dir='cache')
def load_shyamupa_t2id(filepath, filterkeys=None):
    if filterkeys is None:
        d = {}
        with open(filepath, 'r') as fin:
            for l in fin:
                l = l.strip()
                if len(l) == 0:
                    continue
                v, k, _ = l.split('\t')
                d[k] = v
        return d
    else:
        def gen():
            with open(filepath, 'r') as fin:
                for l in fin:
                    l = l.strip()
                    if len(l) == 0:
                        continue
                    v, k, _ = l.split('\t')
                    if k in filterkeys:
                        yield l
        return gen


def plot_correlation(matrix, classes, title, saveto):
    fig, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)

    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    '''
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if True else 'd'
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")
    '''
    #fig.tight_layout()
    plt.savefig(saveto)


def save_emb_ids(filepath, out_filepath):
    result = set()
    with open(filepath, 'r') as fin, open(out_filepath, 'w') as fout:
        for l in tqdm(fin):
            l = l.split('\t', 1)
            id = l[0]
            result.add(id)
            fout.write(id + '\n')


def read_emb_ids(filepath, filter=False) -> set:
    print('load emb ids ...')
    result = set()
    with open(filepath, 'r') as fin:
        for id in tqdm(fin):
            id = id.strip()
            if filter:
                if id.startswith('<http://www.wikidata.org/entity/') or \
                        id.startswith('<http://www.wikidata.org/prop/direct/'):
                    id = id.rsplit('/', 1)[1][:-1]
                    result.add(id)
            else:
                result.add(id)
    return result


def load_embedding_cache():
    def wrapper(func):
        @functools.wraps(func)
        def new_func(filepath, *args, **kwargs):
            emb_cache = filepath + '.npz'
            emb_id2ind_cache = filepath + '.id2ind'
            if os.path.exists(emb_cache) and os.path.exists(emb_id2ind_cache):
                print('load emb from cache ...')
                with open(emb_id2ind_cache, 'rb') as fin:
                    emb_id2ind = pickle.load(fin)
                emb = np.load(emb_cache)
            else:
                emb_id2ind, emb = func(filepath, *args, **kwargs)
                print('cache emb ...')
                with open(emb_id2ind_cache, 'wb') as fout:
                    pickle.dump(emb_id2ind, fout)
                with open(emb_cache, 'wb') as fout:
                    np.save(fout, emb)
            return emb_id2ind, emb
        return new_func
    return wrapper


def load_embedding(filepath, debug=False, emb_size=None) -> Tuple[Dict[str, int], np.ndarray]:
    print('load emb ...')
    id2ind = {}
    emb = []
    with open(filepath, 'r') as fin:
        for i, l in tqdm(enumerate(fin)):
            l = l.split('\t', 1)
            id2ind[l[0]] = i
            if debug:
                emb.append([0.1] * emb_size)
            else:
                l = list(map(float, l[1].split('\t')))
                if emb_size and len(l) != emb_size:
                    raise ValueError('emb dim incorrect')
                emb.append(l)
    return id2ind, np.array(emb, dtype=np.float32)


@load_embedding_cache()
def read_embeddings_from_text_file(filepath: str,
                                   debug: bool = False,
                                   emb_size: int = None,
                                   first_line: bool = False,
                                   use_padding: bool = False) -> Tuple[Dict[str, int], np.ndarray]:
    print('load emb from {} ...'.format(filepath))
    id2ind = defaultdict(lambda: len(id2ind))
    emb = []
    if use_padding:  # put padding at the first position
        _ = id2ind[PADDING]
    with EmbeddingsTextFile(filepath) as embeddings_file:
        if first_line:
            embeddings_file.readline()
        for i, line in tqdm(enumerate(embeddings_file)):
            token = line.split('\t', 1)[0]
            _ = id2ind[token]
            if debug:
                emb.append([0.1] * emb_size)
            else:
                l = np.asarray(line.rstrip().split('\t')[1:], dtype='float32')
                if emb_size and len(l) != emb_size:
                    raise ValueError('emb dim incorrect')
                emb.append(l)
    if use_padding:  # put padding at the first position
        if emb_size:
            emb.insert(0, [0] * emb_size)
        else:
            emb.insert(0, [0] * len(emb[0]))
    return id2ind, np.array(emb, dtype=np.float32)


def filer_embedding(in_filepath: str,
                    out_filepath: str,
                    ids: set):
    print('filter emb ...')
    seen_ids = set()
    with open(in_filepath, 'r') as fin, open(out_filepath, 'w') as fout:
        num_entity, num_prop, dim = fin.readline().split('\t')
        for i, l in tqdm(enumerate(fin)):
            ls = l.split('\t', 1)
            id = ls[0]
            if id.startswith('<http://www.wikidata.org/entity/') or \
                    id.startswith('<http://www.wikidata.org/prop/direct/'):
                id = id.rsplit('/', 1)[1][:-1]
            if id not in ids:
                continue
            seen_ids.add(id)
            fout.write(id + '\t' + ls[1])
            if len(seen_ids) >= len(ids):
                break


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory


    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)


    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value


    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()


    def copy(self):
        return self.__copy__()


    def __copy__(self):
        return type(self)(self.default_factory, self)


    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))


    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))