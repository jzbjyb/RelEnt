from typing import Tuple, List, Dict
from collections import defaultdict
import numpy as np
from random import shuffle
from operator import itemgetter
import json
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from .property import read_subprop_file, get_is_sibling, get_all_subtree, get_is_parent, get_is_ancestor


subprops = read_subprop_file('data/subprops.txt')
subtrees, _ = get_all_subtree(subprops)
is_sibling = get_is_sibling(subprops)
is_parent = get_is_parent(subprops)
is_ancestor = get_is_ancestor(subtrees)


def get_ranks(ranks: Dict[str, List[Tuple[str, float]]], is_parent, is_ancestor):
    def get_rel(parent, child):
        if (parent, child) in is_parent:
            return 'parent'
        elif (parent, child) in is_ancestor:
            return 'ancestor'
        else:
            return '*'
    new_ranks: Dict[str, List[Tuple[str, float, str]]] = {}
    for query, docs in ranks.items():
        new_ranks[query] = [(doc, score, get_rel(doc, query)) for doc, score in docs]
    return new_ranks


class AnalogyEval():
    ''' Analogy Evaluation '''

    def __init__(self,
                 subprop_file: str,
                 method: str = 'sibling',
                 metric: str = 'accuracy',
                 reduction: str = 'sample',
                 prop_set: set = None,
                 debug: bool = False):

        self.subprops = read_subprop_file(subprop_file)
        self.pid2plabel = dict(p[0] for p in self.subprops)
        self.is_sibling = get_is_sibling(self.subprops)
        self.is_parent = get_is_parent(self.subprops)
        self.subtrees, _ = get_all_subtree(self.subprops)

        assert method in {'sibling', 'parent'}
        self.method = method
        if method == 'sibling':
            self.is_true = self.is_sibling
        elif method == 'parent':
            self.is_true = self.is_parent

        assert metric in {'accuracy', 'auc_map', 'correct_position'}
        # 'accuracy': property pair level accuracy
        # 'auc_map': auc and ap for each ranking of property
        # 'correct_position': whether the property goes in the correct position
        self.metric = metric

        assert reduction in {'sample', 'property'}
        # 'sample': each sample/occurrence of a property is an evaluation unit
        # 'property': each property is an evaluation unit
        self.reduction = reduction

        # only evaluate on these properties
        self.prop_set = prop_set

        # whether to use debug mode
        self.debug = debug

        self._property_accuracy_cache = {}


    def collect_property_score(self, predictions: List[Tuple[str, str, float]]) -> Dict[Tuple[str, str], float]:
        prop_score = defaultdict(lambda: [])
        for prop1, prop2, p in predictions:
            prop_score[(prop1, prop2)].append(p)
        prop_score = dict((k, np.mean(v)) for k, v in prop_score.items())
        return prop_score


    def property_accuracy_comp(self, result: Dict[str, Tuple[float, bool]]):
        for k in result:
            if k in self._property_accuracy_cache and self._property_accuracy_cache[k][1] != result[k][1]:
                print('{}: {} -> {}'.format(k, self._property_accuracy_cache[k], result[k]))
        self._property_accuracy_cache = result


    def property_accuracy(self, predictions: List[Tuple[str, str, float]]):
        prop_score = self.collect_property_score(predictions)

        correct, wrong = 0, 0
        tp, tn, fp, fn = [0] * 4
        tp_, tn_, fp_, fn_ = [], [], [], []
        for k in prop_score:
            v = prop_score[k]
            if v >= 0.5 and k in self.is_true:
                tp += 1
                correct += 1
                prop_score[k] = (prop_score[k], True)
                tp_.append(k)
            elif v < 0.5 and k not in self.is_true:
                tn += 1
                correct += 1
                prop_score[k] = (prop_score[k], True)
                tn_.append(k)
            elif v >= 0.5 and k not in self.is_true:
                fp += 1
                wrong += 1
                prop_score[k] = (prop_score[k], False)
                fp_.append(k)
            else:
                fn += 1
                wrong += 1
                prop_score[k] = (prop_score[k], False)
                fn_.append(k)

        if self.debug:
            print('predictions changes:')
            self.property_accuracy_comp(prop_score)
            def format_prop(prop_li):
                prop_li = [(p1 + ': ' + self.pid2plabel[p1],
                            p2 + ': ' + self.pid2plabel[p2]) for p1, p2 in prop_li]
                shuffle(prop_li)
                return prop_li
            def pair_set_join(pairs, pset):
                return [(p1, p2) for p1, p2 in pairs if p1 in pset and p2 in pset]
            by_tree = [pair_set_join(tp_, st.nodes) for st in self.subtrees]
            json.dump({'tp': format_prop(tp_), 'tn': format_prop(tn_),
                       'fp': format_prop(fp_), 'fn': format_prop(fn_),
                       'by_tree': by_tree}, open('data/result.json', 'w'))

        return correct / (correct + wrong + 1e-10), (tp, tn, fp, fn)


    def sample_accuracy(self, predictions: List[Tuple[str, str, float]]):
        correct, wrong = 0, 0
        for prop1, prop2, p in predictions:
            if (p >= 0.5 and (prop1, prop2) in self.is_true) or \
                    (p < 0.5 and (prop1, prop2) not in self.is_true):
                correct += 1
            else:
                wrong += 1
        return correct / (correct + wrong + 1e-10)


    def property_auc_map(self, predictions: List[Tuple[str, str, float]]):
        # get property average score
        prop_score = self.collect_property_score(predictions)

        # get ranking
        ranks = defaultdict(lambda: {'scores': [], 'labels': []})
        for (prop1, prop2), p in prop_score.items():
            if self.prop_set is None or prop1 in self.prop_set:
                ranks[prop1]['scores'].append(p)
                ranks[prop1]['labels'].append(int((prop1, prop2) in self.is_true))
            if self.prop_set is None or prop2 in self.prop_set:
                ranks[prop2]['scores'].append(p)
                ranks[prop2]['labels'].append(int((prop1, prop2) in self.is_true))

        # compute score
        auc_dict, ap_dict = {}, {}
        for i, p in enumerate(ranks):
            if np.sum(ranks[p]['labels']) == 0:  # skip rankings without positive documents
                continue
            # compute auc
            pre, rec, thres = precision_recall_curve(ranks[p]['labels'], ranks[p]['scores'])
            auc_dict[p] = auc(rec, pre)
            # compute average precision
            ap_dict[p] = average_precision_score(ranks[p]['labels'], ranks[p]['scores'])

        if self.debug:
            sort_by_ap = list(sorted(ap_dict.items(), key=lambda x: -x[1]))
            print('top prop: {}'.format(sort_by_ap[:5]))
            print('bottom prop: {}'.format(sort_by_ap[-5:]))

        return {'count': len(auc_dict),
                'AUC': np.mean(list(auc_dict.values())),
                'MAP': np.mean(list(ap_dict.values()))}


    def property_correct_position(self, predictions: List[Tuple[str, str, float]]):
        if self.method == 'parent':
            prop_score = self.collect_property_score(predictions)
            child2parents = defaultdict(lambda: [])
            for (p, c), score in prop_score.items():
                child2parents[c].append((p, score, (p, c) in self.is_true))
            acc = [int(sorted(child2parents[c], key=lambda x: -x[1])[0][2]) for c in child2parents]
            return np.mean(acc), len(acc)
        else:
            raise NotImplementedError


    def eval(self, predictions: List[Tuple[str, str, float]]):
        '''
        evaluate the predictions,
        tuple is [property1, property2, probability of being analogous].
        '''
        return getattr(self, self.reduction + '_' + self.metric)(predictions)


    def eval_by(self,
                reduction: str,
                metric: str,
                predictions: List[Tuple[str, str, float]]):
        '''
        evaluate the predictions,
        tuple is [property1, property2, probability of being analogous].
        '''
        return getattr(self, reduction + '_' + metric)(predictions)


def accuracy_nway(predictions: List[Tuple[str, np.ndarray, int]],
                  method='macro',
                  agg='product',
                  ind2label=None,
                  topk=1,
                  use_average=False,
                  num_classes=0):
    if topk is None:
        topk = num_classes
    assert method in {'macro', 'micro'}
    pid2acc = defaultdict(lambda: [])
    corr, total = 0, 0
    ranks: Dict[str, List] = {}
    label2ind = dict((v, k) for k, v in ind2label.items())
    pred_labels, real_labels = [], []
    label2pred: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(num_classes))
    for pid, logits, label in predictions:
        if pid in label2ind:
            logits[label2ind[pid]] = -np.inf
        max_logits = np.max(logits)
        exp_logits = np.exp(logits - max_logits)
        label2pred[label] += exp_logits / np.sum(exp_logits)
        if topk == 1:
            ind = [np.argmax(logits)]
        else:
            ind = np.argsort(-logits)[:topk]
        ranks[pid] = [(ind2label[i], logits[i]) for i in ind]
        pred_labels.append(ind[0])
        real_labels.append(label)
        c = 0
        for i in range(min(len(ind), topk)):
            if ind[i] == label:  # TODO: relation with multiple parents
                c = 1
                break
        pid2acc[pid].append(c)
        total += 1
        corr += c

    if use_average:
        method = 'macro'
        total, corr = 0, 0
        ranks: Dict[str, List] = {}
        for pid, logits, label in predictions:
            ind = np.argsort(-label2pred[label])
            ranks[pid] = [(ind2label[i], label2pred[label][i]) for i in ind]
            total += 1
            corr += int(ind[0] == label)

    if method == 'macro':
        acc_per_prop = sorted([(k, np.mean(v)) for k, v in pid2acc.items()], key=lambda x: -x[1])
        acc = np.mean(list(map(itemgetter(1), acc_per_prop)))
        # TODO: to be consistent with overlap methods?
        #print(np.mean([v[np.random.choice(len(v), 1)[0]] for k, v in pid2acc.items()]), len(acc_per_prop))
    elif method == 'micro':
        acc = corr / total
    return acc, ranks, pred_labels, real_labels


def accuracy_pointwise(predictions: List[Tuple[str, str, float, int]], method='macro', agg='product', **kwargs):
    # 'max': whether the class with max prob is parent
    # 'product': whether the parent is selected and non-ancestors are not selected
    # 'rank': whether ancestors get higher scores than non-ancestors
    # 'no': no aggregation
    assert agg in {'product', 'max', 'rank', 'no'}
    # TODO only have macro version
    child2ancestor = defaultdict(lambda: defaultdict(list))
    for parent, child, logits, label in predictions:
        c = (logits >= 0.5 and label) or (logits < 0.5 and not label)
        child2ancestor[child][parent].append(logits)
    # ensemble predictions
    for child in child2ancestor:
        for parent in child2ancestor[child]:
            child2ancestor[child][parent] = np.mean(child2ancestor[child][parent])  # TODO: this is ensemble
    # get ranking for error analysis
    ranks = {}
    for child in child2ancestor:
        ranks[child] = sorted(child2ancestor[child].items(), key=lambda x: -x[1])
        ranks[child] = [(parent, pred) for parent, pred in ranks[child]]
    # eval
    eval_result = []
    corr, total = 0, 0
    if agg == 'product':
        for child in child2ancestor:
            c = 0  # 0 is correct
            for parent, pred in child2ancestor[child].items():
                if (parent, child) in is_parent:  # TODO: relation with multiple parents
                    if pred < 0.5:
                        c = 1  # not find parent
                        break
                elif (parent, child) not in is_ancestor:
                    if pred >= 0.5:
                        c = 2  # find fake parent
                        break
            corr += c == 0
            total += 1
            eval_result.append((child, c))
    elif agg == 'max':
        for child in child2ancestor:
            c = 0  # 0 is correct
            scores = sorted(child2ancestor[child].items(), key=lambda x: -x[1])
            if (scores[0][0], child) not in is_parent:
                c = 1  # not find parent
            corr += c == 0
            total += 1
            eval_result.append((child, c))
        # TODO: add mrr
        '''
        for child in child2ancestor:
            rr = 0
            scores = sorted(child2ancestor[child].items(), key=lambda x: -x[1])
            for i in range(len(scores)):
                if (scores[i][0], child) in is_parent:
                    rr = 1 / (i + 1)
                    break
            corr += rr
            total += 1
            eval_result.append((child, corr))
        '''
    elif agg == 'rank':
        for child in child2ancestor:
            c = 0  # 0 is correct
            scores = sorted(child2ancestor[child].items(), key=lambda x: -x[1])
            found_non_ancestors = False
            for parent, score in scores:
                if (parent, child) not in is_ancestor:
                    found_non_ancestors = True
                elif found_non_ancestors:
                    c = 1  # ancestors are ranked behind non-ancestors
                    break
            corr += c == 0
            total += 1
            eval_result.append((child, c))
    elif agg == 'no':
        for parent, child, logits, label in predictions:
            c = (logits >= 0.5 and label) or (logits < 0.5 and not label)
            corr += int(c)
            total += 1
    return corr / total, ranks


def rank_to_csv(ranks: Dict[str, List], filepath: str, key2name: Dict[str, str] = None,
                simple_save=False, label2ind=None, topk=10):
    def doc_formatter(docid: str, score: float, comment: str):
        if key2name is not None:
            docid = key2name[docid]
        return ' '.join(['"' + docid + '"', '{:.2f}'.format(score), comment])
    max_num_docs = np.max([len(r) for q, r in ranks.items()])
    max_num_docs = min(max_num_docs, topk)
    with open(filepath, 'w') as fout:
        if not simple_save:
            fout.write('query,' + ','.join(map(lambda x: 'pos_' + str(x), range(max_num_docs))) + '\n')
        for q, r in sorted(ranks.items(), key=itemgetter(0)):
            if not simple_save:
                if key2name is not None:
                    q = key2name[q]
                fout.write('"{}"'.format(q) + ',')
                fout.write(','.join(map(lambda x: doc_formatter(*x), r[:max_num_docs])))
            else:
                li = [q] + [label2ind[docid] for docid, _, _ in r]
                fout.write(','.join(li))
            fout.write('\n')
