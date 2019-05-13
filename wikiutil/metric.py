from typing import Tuple, List
from collections import defaultdict
import numpy as np
from .property import read_subprop_file, get_is_sibling


class AnalogyEval():
    ''' Analogy Evaluation '''

    def __init__(self,
                 subprop_file: str,
                 method: str = 'accuracy',
                 reduction: str = 'sample'):

        self.subprops = read_subprop_file(subprop_file)
        self.is_sibling = get_is_sibling(self.subprops)

        assert method in {'accuracy'}
        # 'accuracy': property pair level accuracy
        self.method = method

        assert reduction in {'sample', 'property'}
        # 'sample': each sample/occurrence of a property is an evaluation unit
        # 'property': each property is an evaluation unit
        self.reduction = reduction


    def property_accuracy(self, predictions: List[Tuple[str, str, float]]):
        result = defaultdict(lambda: [])
        for prop1, prop2, p in predictions:
            result[(prop1, prop2)].append(p)
        result = dict((k, np.mean(v)) for k, v in result.items())
        correct, wrong = 0, 0
        for k, v in result.items():
            if (v >= 0.5 and k in self.is_sibling) or \
                    (v < 0.5 and k not in self.is_sibling):
                correct += 1
            else:
                wrong += 1
        return correct / (correct + wrong + 1e-10)


    def eval(self, predictions: List[Tuple[str, str, float]]):
        '''
        evaluate the predictions,
        tuple is [property1, property2, probability of being analogous].
        '''
        return getattr(self, self.reduction + '_' + self.method)(predictions)




