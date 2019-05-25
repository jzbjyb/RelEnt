from typing import Tuple, List, Dict
from collections import defaultdict
import numpy as np
from random import shuffle
import json
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from .property import read_subprop_file, get_is_sibling, get_all_subtree, get_is_parent


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


def accuray(predictions: List[Tuple[str, List, int]]):
    corr, total = 0, 0
    for pid, logits, label in predictions:
        total += 1
        if np.argmax(logits) == label:
            corr += 1
    return corr / total