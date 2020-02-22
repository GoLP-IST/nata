# -*- coding: utf-8 -*-
from typing import TypeVar
from typing import Union

import attr
import numpy as np

T = TypeVar("T")


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
    for attrs_attr in cls.__dict__["__attrs_attrs__"]:
        if attrs_attr.init:
            kwargs_sel.append(attrs_attr.name)

    # build filtered kwargs
    kwargs_flt = {}
    for attrib in kwargs_sel:
        try:
            # TODO: make this use defaults - most likely it can be None @fabio
            #       - if so, try and except can be removed
            #       - else KeyError should be excepted
            prop = kwargs.pop(attrib)
            kwargs_flt[attrib] = prop
        except:  # noqa: E722
            continue

    return kwargs_flt


def attrib_equality(some: T, other: T, props_to_check: Union[str, tuple]):
    if isinstance(props_to_check, str):
        props_to_check = props_to_check.replace(" ", "").split(",")

    for prop in props_to_check:
        if getattr(some, prop) != getattr(other, prop):
            return False
    return True
