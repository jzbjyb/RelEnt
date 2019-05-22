from typing import Tuple, List, Dict
from collections import defaultdict
import numpy as np
from random import shuffle
import json
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from .property import read_subprop_file, get_is_sibling, get_all_subtree


class AnalogyEval():
    ''' Analogy Evaluation '''

    def __init__(self,
                 subprop_file: str,
                 method: str = 'accuracy',
                 reduction: str = 'sample',
                 prop_set: set = None,
                 debug: bool = False):

        self.subprops = read_subprop_file(subprop_file)
        self.pid2plabel = dict(p[0] for p in self.subprops)
        self.is_sibling = get_is_sibling(self.subprops)
        self.subtrees, _ = get_all_subtree(self.subprops)

        assert method in {'accuracy', 'auc_map'}
        # 'accuracy': property pair level accuracy
        # 'auc_map': auc and ap for each ranking of property
        self.method = method

        assert reduction in {'sample', 'property'}
        # 'sample': each sample/occurrence of a property is an evaluation unit
        # 'property': each property is an evaluation unit
        self.reduction = reduction

        # only evaluate on these properties
        self.prop_set = prop_set

        # whether to use debug mode
        self.debug = debug

        self._property_accuracy_cache = {}


    def property_accuracy_comp(self, result: Dict[str, Tuple[float, bool]]):
        for k in result:
            if k in self._property_accuracy_cache and self._property_accuracy_cache[k][1] != result[k][1]:
                print('{}: {} -> {}'.format(k, self._property_accuracy_cache[k], result[k]))
        self._property_accuracy_cache = result


    def property_accuracy(self, predictions: List[Tuple[str, str, float]]):
        result = defaultdict(lambda: [])
        for prop1, prop2, p in predictions:
            result[(prop1, prop2)].append(p)
        result = dict((k, np.mean(v)) for k, v in result.items())
        correct, wrong = 0, 0
        tp, tn, fp, fn = [0] * 4
        tp_, tn_, fp_, fn_ = [], [], [], []
        for k in result:
            v = result[k]
            if v >= 0.5 and k in self.is_sibling:
                tp += 1
                correct += 1
                result[k] = (result[k], True)
                tp_.append(k)
            elif v < 0.5 and k not in self.is_sibling:
                tn += 1
                correct += 1
                result[k] = (result[k], True)
                tn_.append(k)
            elif v >= 0.5 and k not in self.is_sibling:
                fp += 1
                wrong += 1
                result[k] = (result[k], False)
                fp_.append(k)
            else:
                fn += 1
                wrong += 1
                result[k] = (result[k], False)
                fn_.append(k)

        if self.debug:
            print('predictions changes:')
            self.property_accuracy_comp(result)
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
            if (p >= 0.5 and (prop1, prop2) in self.is_sibling) or \
                    (p < 0.5 and (prop1, prop2) not in self.is_sibling):
                correct += 1
            else:
                wrong += 1
        return correct / (correct + wrong + 1e-10)


    def property_auc_map(self, predictions: List[Tuple[str, str, float]]):
        # get property average score
        prop_score = defaultdict(lambda: [])
        for prop1, prop2, p in predictions:
            prop_score[(prop1, prop2)].append(p)
        prop_score = dict((k, np.mean(v)) for k, v in prop_score.items())

        # get ranking
        ranks = defaultdict(lambda: {'scores': [], 'labels': []})
        for (prop1, prop2), p in prop_score.items():
            if self.prop_set is None or prop1 in self.prop_set:
                ranks[prop1]['scores'].append(p)
                ranks[prop1]['labels'].append(int((prop1, prop2) in self.is_sibling))
            if self.prop_set is None or prop2 in self.prop_set:
                ranks[prop2]['scores'].append(p)
                ranks[prop2]['labels'].append(int((prop1, prop2) in self.is_sibling))

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


    def eval(self, predictions: List[Tuple[str, str, float]]):
        '''
        evaluate the predictions,
        tuple is [property1, property2, probability of being analogous].
        '''
        return getattr(self, self.reduction + '_' + self.method)(predictions)


    def eval_by(self,
                reduction: str,
                method: str,
                predictions: List[Tuple[str, str, float]]):
        '''
        evaluate the predictions,
        tuple is [property1, property2, probability of being analogous].
        '''
        return getattr(self, reduction + '_' + method)(predictions)
