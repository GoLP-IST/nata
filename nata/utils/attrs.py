# -*- coding: utf-8 -*-
from typing import Any
from typing import Optional
from typing import TypeVar
from typing import Union

import attr
import numpy as np

T = TypeVar("T")


@attr.s
class _SubDtypeValidator:
    type_: type = attr.ib()

    def __call__(self, instance, attribute, value):
        if not np.issubdtype(type(value), self.type_):
            raise TypeError(
                f"Attribute '{attribute.name}' of {instance.__class__} "
                + f"must be of type {self.type_} not {type(value)}"
            )


def subdtype_of(type_):
    return _SubDtypeValidator(type_)


@attr.s
class _ArrayValidator:
    dtype: Optional[type] = attr.ib(default=None)
    subclass: Optional[type] = attr.ib(default=None)

    def __call__(self, instance, attribute, value):
        if isinstance(value, np.ndarray):
            value_type = value.dtype
        else:
            value_type = type(value)

        if self.dtype is not None and not np.issubdtype(value_type, self.dtype):
            # TODO: add better error message
            raise TypeError(
                f"Attribute '{attribute.name}' of {instance.__class__} "
                + f"must have dtype {self.dtype}"
            )

        if self.subclass is not None and not np.issubclass_(
            type(value), self.subclass
        ):
            # TODO: add better error message
            raise TypeError(
                f"Attribute '{attribute.name}' of {instance.__class__} "
                + f"must be of subclass {self.subclass}"
            )


def array_validator(dtype=None, subclass=None):
    return _ArrayValidator(dtype, subclass)


def have_attr(*args):
    if len(args) == 0:
        return False

    for a in args:
        if not attr.has(a.__class__):
            return False

    return True


def attrib_equality(
    some: T, other: Union[T, Any], props_to_check: Union[str, tuple] = None
):
    # if props_to_check is None, inspect attributes based on attrs and only for
    # some. attributes in other and their equality check is discarded
    # TODO: deep check
    if props_to_check is None:
        for key, attrib in attr.fields_dict(some.__class__).items():
            if attrib.eq:
                if not hasattr(other, attrib.name):
                    return False

                some_attrib = getattr(some, attrib.name)
                other_attrib = getattr(other, attrib.name)
                # check if attributes have attrs themselves and use
                # attrib_equality recursively else use equality
                if have_attr(some_attrib, other_attrib):
                    if not attrib_equality(some_attrib, other_attrib):
                        return False
                else:
                    if some_attrib != other_attrib:
                        return False

        return True
    else:
        if isinstance(props_to_check, str):
            props_to_check = props_to_check.replace(" ", "").split(",")

        for prop in props_to_check:
            if getattr(some, prop) != getattr(other, prop):
                return False
        return True


def is_identifier(instance, attribute, value):
    if not value.isidentifier():
        raise ValueError(
            f"attribute {attribute.name} has an invalid string '{value}'"
        )


def location_exists(instance, attribute, value):
    if not value.exists():
        raise ValueError(f"Path '{value}' does not exist!")
