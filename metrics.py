"""
Source: https://sklearn-crfsuite.readthedocs.io/en/latest/_modules/sklearn_crfsuite/metrics.html
"""

from functools import wraps
from itertools import chain

def flatten(y):
    """
    Flatten a list of lists.
    >>> flatten([[1,2], [3,4]])
    [1, 2, 3, 4]
    """
    return list(chain.from_iterable(y))

def _flattens_y(func):
    @wraps(func)
    def wrapper(y_true, y_pred, *args, **kwargs):
        y_true_flat = flatten(y_true)
        y_pred_flat = flatten(y_pred)
        return func(y_true_flat, y_pred_flat, *args, **kwargs)
    return wrapper

@_flattens_y
def flat_classification_report(y_true, y_pred, labels=None, **kwargs):
    """
    Return classification report for sequence items.
    """
    from sklearn import metrics
    return metrics.classification_report(y_true, y_pred, labels, **kwargs)


@_flattens_y
def flat_f1_score(y_true, y_pred, **kwargs):
    """
    Return F1 score for sequence items.
    """
    from sklearn import metrics
    return metrics.f1_score(y_true, y_pred, **kwargs)
