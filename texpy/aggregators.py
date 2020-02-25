"""
Data aggregation routines.

This package contains a variety of routines that can be used to combine
annotations from multiple workers. These routines will typically be used
by :func:`texpy.experiment.TaskHelper.aggregate_responses`.
"""
from typing import List, TypeVar, Callable, Optional, Tuple, NamedTuple
from collections import Counter
import numpy as np
import scipy.stats as scstats

from .util import Span, WeightedSpan, collapse_spans

T = TypeVar("T")
U = TypeVar("U")


Span = Tuple[int, int]


def majority_interval(lst: List[List[Span]]) -> List[Span]:
    """
    Aggregates a list of intervals

    Example (annotations are the underlines):
    This is an example
    ----
    ----    ----------
            ----------
    ------------------

    would result in
    This is an example
    ----    ----------
    """
    # 1. Collapse spans into non-overlapping sets weighted by frequency
    canonical_spans: List[WeightedSpan] = collapse_spans([
        span for spans in lst for span in spans])

    # 3. Filter to only majoritarian spans
    ret = [(span.begin, span.end) for span in canonical_spans if span.count >= len(lst)/2]

    # 4. Collapse adjacent intervals that were selected
    for i in range(len(ret)-1, 0, -1):
        span, prev_span = ret[i], ret[i-1]
        if prev_span[1] == span[0]:
            ret.pop(i)
            ret[i-1] = (prev_span[0], span[1])

    return ret


def test_majority_interval():
    # This is an example
    # ----
    # ----    ----------
    #         ----------
    # ------------------
    assert [(0, 4), (8, 18)] == majority_interval([
        [(0, 4),],
        [(0, 4), (8, 18)],
        [(8, 18),],
        [(0, 18)],
        ])

    assert [(0, 4), (8, 18)] == majority_interval([
        [(0, 4),],
        [(8, 18),],
        [(0, 4), (8, 18)],
        [(0, 18)],
        ])

    # This is an example
    # ----
    # ----    ----------
    #         ----------
    # --------------    
    #   -------------   
    assert [(0, 4), (8, 16)] == majority_interval([
        [(0, 4),],
        [(0, 4), (8, 18)],
        [(8, 18),],
        [(0, 15)],
        [(2, 16)],
        ])


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
           'transpose', 'apply_list', 'majority_vote', 'majority_interval']
