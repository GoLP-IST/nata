# -*- coding: utf-8 -*-
from typing import Tuple

import attr
import numpy as np
from attr import converters
from attr.validators import in_
from attr.validators import instance_of

from nata.utils.attrs import attrib_equality
from nata.utils.attrs import subdtype_of

_incomparable = {"order": False, "eq": False}


@attr.s(slots=True, eq=False, order=False)
class UnnamedAxis:
    _data: np.ndarray = attr.ib(
        converter=converters.optional(np.array), repr=False
    )

    _data_ndim: Tuple[int] = attr.ib(init=False, repr=False)

    def __attrs_post_init__(self):
        self._data_ndim = self._data.ndim

    def __array__(self, dtype=None):
        return self._data.astype(dtype)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return attrib_equality(self, other, "_data_ndim")

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __len__(self):
        if self._data_ndim == self.ndim:
            return 1
        else:
            return len(self.data)

    def __iter__(self):
        if len(self) == 0:
            yield self.__class__(self._data, self.name, self.label, self.unit)
        else:
            for d in self._data:
                yield self.__class__(d, self.name, self.label, self.unit)

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def data(self):
        return self._data

    def _check_appendability(self, other):
        if not isinstance(other, Axis):
            raise TypeError("Can only append Axis objects to Axis")

        if not self == other:
            raise ValueError(f"{other} can not be append to {self}")

    def append(self, other):
        self._check_appendability(other)
        self._data = np.hstack([self.data, other.data])


@attr.s(slots=True, eq=False, order=False)
class Axis(UnnamedAxis):
    name: str = attr.ib(validator=subdtype_of(np.str_))
    label: str = attr.ib(validator=subdtype_of(np.str_))
    unit: str = attr.ib(validator=subdtype_of(np.str_))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return attrib_equality(self, other, "name, label, unit, _data_ndim")


@attr.s(slots=True, eq=False, order=False)
class IterationAxis(Axis):
    name: str = attr.ib(default="iteration", validator=subdtype_of(np.str_))
    label: str = attr.ib(default="iteration", validator=subdtype_of(np.str_))
    unit: str = attr.ib(default="", validator=subdtype_of(np.str_))


@attr.s(slots=True, eq=False, order=False)
class TimeAxis(Axis):
    name: str = attr.ib(default="time", validator=subdtype_of(np.str_))
    label: str = attr.ib(default="time", validator=subdtype_of(np.str_))
    unit: str = attr.ib(default="", validator=subdtype_of(np.str_))


@attr.s(slots=True, eq=False, order=False)
class GridAxis(Axis):
    axis_length: int = attr.ib(validator=instance_of(int))
    axis_type: str = attr.ib(
        default="linear",
        validator=[subdtype_of(np.str_), in_(("linear", "logarithmic"))],
    )

    # we might be able to remove this and just use the 'Axis' definition
    # -> has to be tried but I think we can just use attrs to pass the arguments
    def __iter__(self):
        if len(self) == 1:
            yield self.__class__(
                self._data,
                self.name,
                self.label,
                self.unit,
                self.axis_length,
                self.axis_type,
            )
        else:
            for d in self._data:
                yield self.__class__(
                    d,
                    self.name,
                    self.label,
                    self.unit,
                    self.axis_length,
                    self.axis_type,
                )

    @property
    def shape(self):
        if len(self):
            return (len(self), self.axis_length)
        else:
            return (self.axis_length,)

    def _get_axis_values(self, min_, max_, N):
        if self.axis_type == "logarithmic":
            min_, max_ = np.log10((min_, max_))
            return np.logspace(min_, max_, N)
        else:
            return np.linspace(min_, max_, N)

    def __array__(self, dtype=None):
        if len(self) == 1:
            return self._get_axis_values(
                self._data[0], self._data[1], self.axis_length
            )
        else:
            values = np.empty((len(self), self.axis_length))
            for i, (min_, max_) in enumerate(self._data):
                values[i] = self._get_axis_values(min_, max_, self.axis_length)

            return values

    def append(self, other):
        self._check_appendability(other)
        self._data = np.vstack([self.data, other.data])
