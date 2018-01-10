"""
Utilities
"""

import json
import logging
from typing import Dict
from .quality_control import QualityControlDecision

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o: object) -> Dict:
        if isinstance(o, QualityControlDecision):
            return {
                    "_type": "QualityControlDecision",
                    "_fields": {
                        "should_approve": o.should_approve,
                        "short_reason": o.short_reason,
                        "reason": o.reason,
                        "qualification_update": o.qualification_update,
                    }
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
        ret = input(prompt + str(options) + ": ").strip()
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


def sanitize(obj):
    if isinstance(obj, list):
        return [sanitize(obj_) for obj_ in obj]
    elif isinstance(obj, dict):
        return {key: sanitize(value) for key, value in obj.items() if not key.startswith("_")}
    else:
        return obj
