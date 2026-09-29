"""Content identity for diagnostic inputs; host hashing is evidence I/O only."""
from dataclasses import fields,is_dataclass
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import torch

from ..compute import to_host


def content_digest(value):
    digest = sha256()
    def add(item):
        if torch.is_tensor(item) or isinstance(item,np.ndarray) or hasattr(item,'__cuda_array_interface__'):
            array = np.asarray(to_host(item))
            if array.dtype.hasobject:
                raise ValueError('Object array in diagnostic binding')
            digest.update(json.dumps(['array',str(array.dtype),list(array.shape)]).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        elif is_dataclass(item):
            add({f.name:getattr(item,f.name) for f in fields(item)})
        elif isinstance(item,SimpleNamespace):
            add(vars(item))
        elif isinstance(item,dict):
            digest.update(b'{')
            for key in sorted(item):
                if not isinstance(key,str): raise ValueError('Non-string binding key')
                add(key);add(item[key])
            digest.update(b'}')
        elif isinstance(item,(list,tuple)):
            digest.update(b'[')
            for child in item: add(child)
            digest.update(b']')
        elif isinstance(item,np.generic):
            add(item.item())
        elif item is None or type(item) in (bool,int,float,str):
            encoded = json.dumps(item,allow_nan=False).encode()
            digest.update(str(len(encoded)).encode()+b':'+encoded)
        else:
            raise TypeError('Unsupported diagnostic binding: '+type(item).__name__)
    add(value)
    return digest.hexdigest()
