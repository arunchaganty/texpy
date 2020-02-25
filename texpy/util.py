"""
Utilities
"""

import heapq
import json
import logging
from typing import Dict, TypeVar, Tuple, NamedTuple, List
from .quality_control import QualityControlDecision


T = TypeVar("T")


logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o: object) -> Dict:
        if isinstance(o, QualityControlDecision):
            return {
                    "_type": "QualityControlDecision",
                    "_fields": o.asdict(),
            }
        else:
            return super().default(o)

def custom_decoder_object_hook(o: object):
    if "_type" not in o:
        return o
    else:
        if o["_type"] == "QualityControlDecision" and "_fields" in o:
            return QualityControlDecision(**o["_fields"])

def JsonFile(*args, **kwargs):
    def _ret(fname):
        return json.load(open(fname, *args, **kwargs), object_hook=custom_decoder_object_hook)
    return _ret

def load_jsonl(fstream):
    if isinstance(fstream, str):
        with open(fstream) as fstream_:
            return load_jsonl(fstream_)

    return [json.loads(line, object_hook=custom_decoder_object_hook) for line in fstream]

def save_jsonl(fstream, objs):
    if isinstance(fstream, str):
        with open(fstream, "w") as fstream_:
            save_jsonl(fstream_, objs)
        return

    for obj in objs:
        fstream.write(json.dumps(obj, sort_keys=True, cls=CustomJSONEncoder))
        fstream.write("\n")

def first(itable):
    try:
        return next(iter(itable))
    except StopIteration:
        return None

def force_user_input(prompt, options):
    ret = None
    while ret is None:
        ret = input(prompt + "|".join(options) + ": ").strip()
        if ret.lower() not in options:
            ret = None
    return ret

def obj_diff(obj, obj_):
    ret = True
    for k in obj:
        if k not in obj_:
            print("Key {} missing in arg2".format(k))
            ret = False
        if obj[k] != obj_[k]:
            if isinstance(obj[k], dict) and isinstance(obj_[k], dict):
                obj_diff(obj[k], obj_[k])
                ret = False
            else:
                print("{}: args1 has {}, args2 has {}".format(k, obj[k], obj_[k]))
    for k in obj_:
        if k not in obj:
            print("Key {} missing in arg1".format(k))
            ret = False
    return ret


def sanitize(obj: T) -> T:
    """
    Sanitize an object containing dictionaries by removing any entries
    with a key that starts with '_'.
    """
    if isinstance(obj, list):
        return [sanitize(obj_) for obj_ in obj]
    elif isinstance(obj, dict):
        return {key: sanitize(value) for key, value in obj.items() if not key.startswith("_")}
    else:
        return obj


# region: Span utilities
Span = Tuple[int, int]


class WeightedSpan(NamedTuple):
    """
    A span tuple with a weight / count field
    """
    begin: int
    end: int
    count: int = 1


def collapse_spans(lst: List[Span]) -> List[WeightedSpan]:
    """
    Convert a list of spans into non-overlapping versions with weights
    for each overlapping section.
    """
    if not lst: return []

    all_spans = [WeightedSpan(*span) for span in lst]
    heapq.heapify(all_spans)

    # 1. Figure out what the interval spans that we'll count over are
    #    We do this by setting up split points
    canonical_spans = [heapq.heappop(all_spans)]
    while all_spans:
        span = heapq.heappop(all_spans)
        last_span = canonical_spans[-1]
        assert last_span.begin <= span.begin

        # If the spans don't even overlap, we can safely add this to
        # the canonical list.
        if not(last_span.begin < span.end and span.begin < last_span.end):
            canonical_spans.append(span)
        # We now handle the different overlapping cases.
        elif last_span.begin < span.begin:
            # We are going to split last_span and span into two segments
            # each (with one overlapping span) pivoted at span.begin
            # First, we'll update last_span to its new boundary.
            canonical_spans[-1] = WeightedSpan(last_span.begin, span.begin, last_span.count)
            # Then, we'll break last_span by adding [span.begin,
            # last_span.end) to our queue
            heapq.heappush(all_spans, 
                    WeightedSpan(span.begin, last_span.end, last_span.count + span.count))
            # Finally, we'll break span by adding [last_span.end,
            # span.end) to our queue
            heapq.heappush(all_spans, 
                    WeightedSpan(last_span.end, span.end, span.count))
        elif last_span.end < span.end:
            # We are going to split span into two segments pivoted
            # around last_span.end, and increment counts appropriately
            canonical_spans[-1] = WeightedSpan(last_span.begin, last_span.end,
                    last_span.count + span.count)
            # Create a new segment from [last_span.end, span.end)
            heapq.heappush(all_spans,
                    WeightedSpan(last_span.end, span.end, span.count))
        else:
            # We have a complete overlap and are just going to increment
            # counts
            canonical_spans[-1] = WeightedSpan(last_span.begin, last_span.end,
                    last_span.count + span.count)

    return canonical_spans
