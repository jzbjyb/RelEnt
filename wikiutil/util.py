import functools, os
import matplotlib.pyplot as plt
import numpy as np


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