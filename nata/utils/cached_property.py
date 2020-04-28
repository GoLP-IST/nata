# -*- coding: utf-8 -*-
import sys

# cached_property was incorporated in  python 3.8+
if sys.version_info >= (3, 8):
    from functools import cached_property
else:
    cached_property = property

    # something is wrong in the implementation
    # class cached_property:
    #     def __init__(self, func):
    #         self._func = func

    #     def __get__(self, instance, owner=None):
    #         assert instance is not None
    #         ret = \
    #           instance.__dict__[self._func.__name__] = self._func(instance)
    #         return ret
