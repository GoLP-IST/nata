# -*- coding: utf-8 -*-
import attr
import numpy as np


@attr.s
class _SubDtypeValidator:
    numeric_type: type = attr.ib()

    def __call__(self, instance, attribute, value):
        if not np.issubdtype(type(value), self.numeric_type):
            raise TypeError(
                f"'{attribute.name}' must be of type '{self.numeric_type}', "
                + f"but got {type(value)}"
            )


def subdtype_of(type_):
    return _SubDtypeValidator(type_)


def filter_kwargs(cls, **kwargs):
    
    # get selectable kwargs from class definition
    kwargs_sel = []
    for attr in cls.__dict__["__attrs_attrs__"]:
        if attr.init:
            kwargs_sel.append(attr.name)

    # build filtered kwargs
    kwargs_flt = {}
    for attr in kwargs_sel:
        try: 
            prop = kwargs.pop(attr)
            kwargs_flt[attr] = prop
        except:
            continue
        
    return kwargs_flt