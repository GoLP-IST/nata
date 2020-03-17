# -*- coding: utf-8 -*-
from functools import partial
from typing import Any
from typing import Union

import attr
import numpy as np
from attr import converters
from attr.validators import in_
from attr.validators import instance_of

from nata.utils.attrs import array_validator
from nata.utils.attrs import attrib_equality
from nata.utils.attrs import subdtype_of
from nata.utils.formatting import array_format

axis_attrs = partial(attr.s, slots=True, eq=False, repr=False)


@axis_attrs
class UnnamedAxis:
    _data: np.ndarray = attr.ib(converter=np.asanyarray, repr=False, eq=False)

    def __array__(self, dtype=None):
        if dtype:
            return self._data.astype(dtype)
        else:
            return self._data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    def __iter__(self) -> "UnnamedAxis":
        if self.ndim == 0:
            yield self
        else:
            for d in self._data:
                yield self.__class__(data=d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({array_format(self.data)})"

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
        if not isinstance(other, self.__class__):
            raise TypeError("Can only append Axis objects to Axis")

        if not attrib_equality(self, other):
            raise ValueError(f"{other} can not be append to {self}")

    def append(self, other):
        self._check_appendability(other)
        self._data = np.hstack([self.data, other.data])


@axis_attrs
class Axis(UnnamedAxis):
    name: str = attr.ib(validator=subdtype_of(np.str_))
    label: str = attr.ib(validator=subdtype_of(np.str_))
    unit: str = attr.ib(validator=subdtype_of(np.str_))

    # TODO: remove eq -> should be replaced with proper check after
    #       new implementation of attrib_equality
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return attrib_equality(self, other, "name, label, unit, _data_ndim")

    def __iter__(self) -> "Axis":
        if self.ndim == 0:
            yield self
        else:
            for d in self._data:
                yield self.__class__(
                    data=d, name=self.name, label=self.label, unit=self.unit
                )


@axis_attrs
class IterationAxis(Axis):
    name: str = attr.ib(default="iteration", validator=subdtype_of(np.str_))
    label: str = attr.ib(default="iteration", validator=subdtype_of(np.str_))
    unit: str = attr.ib(default="", validator=subdtype_of(np.str_))


@axis_attrs
class TimeAxis(Axis):
    name: str = attr.ib(default="time", validator=subdtype_of(np.str_))
    label: str = attr.ib(default="time", validator=subdtype_of(np.str_))
    unit: str = attr.ib(default="", validator=subdtype_of(np.str_))


@axis_attrs
class GridAxis(Axis):
    axis_length: int = attr.ib(validator=instance_of(int))
    axis_type: str = attr.ib(
        default="linear",
        validator=[subdtype_of(np.str_), in_(("linear", "log"))],
    )
    min_: float = attr.ib(default=None, converter=converters.optional(float))
    max_: float = attr.ib(default=None, converter=converters.optional(float))
    _data: np.ndarray = attr.ib(default=None, repr=False, eq=False)

    def __attrs_post_init__(self):
        if self._data is None and (
            (self.min_ is not None) and (self.max_ is not None)
        ):
            self._data = np.array([self.min_, self.max_])
        self._data_ndim = self._data.ndim

    # we might be able to remove this and just use the 'Axis' definition
    # -> has to be tried but I think we can just use attrs to pass the arguments
    def __iter__(self):
        if len(self) == 1:
            yield self.__class__(
                min_=self._data[0],
                max_=self._data[1],
                name=self.name,
                label=self.label,
                unit=self.unit,
                axis_length=self.axis_length,
                axis_type=self.axis_type,
            )
        else:
            for d in self._data:
                yield self.__class__(
                    min_=d[0],
                    max_=d[1],
                    name=self.name,
                    label=self.label,
                    unit=self.unit,
                    axis_length=self.axis_length,
                    axis_type=self.axis_type,
                )

    @property
    def shape(self):
        if self._data.ndim == 1:
            return (self.axis_length,)
        else:
            return (len(self), self.axis_length)

    def _get_axis_values(self, min_, max_, N):
        if self.axis_type == "log":
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


@axis_attrs
class ParticleQuantity(Axis):
    # _data in this case can be an array directly or can be a something to get
    # the data later
    _data: np.ndarray = attr.ib(converter=np.array, eq=False)
    _dtype: np.dtype = attr.ib(validator=instance_of(np.dtype))
    _len: np.ndarray = attr.ib(
        converter=np.array,
        validator=array_validator(dtype=np.integer),
        eq=False,
    )

    def __array__(self, dtype=None):
        if self._data.dtype == object:
            max_ = np.max(self._len)
            data = np.ma.empty((len(self._data), max_), dtype=self._dtype)
            data.mask = np.ones((len(self._data), max_), dtype=np.bool)

            for i, (d, entries) in enumerate(zip(self._data, self._len)):
                if isinstance(d, np.ndarray):
                    data[i, :entries] = d[:entries]
                else:
                    data[i, :entries] = d.get_data(fields=self.name)

                data.mask[i, entries:] = np.zeros(max_ - entries, dtype=np.bool)

            self._data = data
        return np.squeeze(self._data)

    def __iter__(self):
        if len(self._data) != 1:
            for d, l in zip(self._data, self._len):
                yield self.__class__(
                    data=[d],
                    len=[l],
                    dtype=self._dtype,
                    name=self.name,
                    label=self.label,
                    unit=self.unit,
                )
        else:
            yield self.__class__(
                data=self._data,
                len=self._len,
                dtype=self._dtype,
                name=self.name,
                label=self.label,
                unit=self.unit,
            )

    def __getitem__(self, key):
        raise self.__array__()[key]

    @property
    def shape(self):
        if self._data.dtype == object:
            if len(self._data) == 1:
                return (np.max(self._len),)
            else:
                return (len(self._data), np.max(self._len))
        else:
            return self._data.shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self.shape)

    def append(self, other: Union["ParticleQuantity", Any]):
        self._check_appendability(other)

        self._data = np.array(
            [d for d in self._data] + [d for d in other._data]
        )
        self._len = np.concatenate((self._len, other._len))
