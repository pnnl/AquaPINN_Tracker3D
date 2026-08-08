# -*- coding: utf-8 -*-
import sys
import timeit
from functools import wraps
import logging
logger = logging.getLogger(__name__)

def timing(f):
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = timeit.default_timer()
        result = f(*args, **kwargs)
        te = timeit.default_timer()
        logger.debug("%r took %f s\n" % (f.__name__, te - ts))
        sys.stdout.flush()
        return result
    return wrapper



def treeFunction(fun, **kwargs):
    @wraps(fun)
    def Func(*args):
        out=[]
        for arg in args:
            if isinstance(arg, dict):
                tmp={}
                for key in arg:
                    tmp[key]=Func(arg[key])
                out.append(tmp)
            elif isinstance( arg, (list, tuple)):
                tmp=[ Func(iarg) for iarg in arg]
                out.append(type(arg)(tmp))
            else:
                out.append(fun(arg, **kwargs) if arg is not None else None)
        return out if len(out)>1 else out[0]
    return Func
