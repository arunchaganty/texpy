"""
Various data aggregation routines.
"""
from typing import List, TypeVar, Callable, Optional
from collections import Counter
import numpy as np
import scipy.stats as scstats

T = TypeVar("T")
U = TypeVar("U")


def transpose(lst: List[List[T]]) -> List[List[T]]:
    n = len(lst[0])
    assert all(len(lst_) == n for lst_ in lst), "Expected inputs to have uniform lengths"

    return list(zip(*lst))


def apply_list(fn: Callable[[List[T]], U], lst: List[List[T]]) -> List[U]:
    """
    Applies a single function to every element of a list
    """
    if not lst:
        return []
    return [fn(lst_) for lst_ in transpose(lst)]


def majority_vote(lst: List[T]) -> T:
    """
    Returns the most common value
    """
    (elem, _), = Counter(lst).most_common(1)
    return elem


def _float(fn: Callable[[T], U]) -> Callable[[T], float]:
    return lambda x, *args, **kwargs: float(fn(x, *args, **kwargs))


def _guard(fn: Callable[[T], U]) -> Callable[[T], Optional[U]]:
    return lambda x, *args, **kwargs: fn(x, *args, **kwargs) if x else None


mean = _guard(_float(np.mean))
std = _guard(_float(np.std))
median = _guard(_float(np.median))
median_absolute_deviation = _guard(_float(scstats.median_absolute_deviation))
mode = _guard(_float(scstats.mode))
trim_mean = _guard(_float(scstats.trim_mean))
percentile = _guard(_float(np.percentile))


__all__ = ['mean', 'std', 'median', 'median_absolute_deviation', 'mode', 'trim_mean', 'percentile',
           'transpose', 'apply_list', 'majority_vote', ]
