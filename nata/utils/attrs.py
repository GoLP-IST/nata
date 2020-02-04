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
