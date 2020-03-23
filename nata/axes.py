# -*- coding: utf-8 -*-
from functools import partial
from typing import Any
from typing import Optional
from typing import Union

import attr
import numpy as np
from attr.validators import in_

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
            raise ValueError(f"{other} can not append to {self}")

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
        return attrib_equality(self, other, "name, label, unit")

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
    _min: Optional[np.ndarray] = attr.ib(converter=np.asanyarray, eq=False)
    _max: Optional[np.ndarray] = attr.ib(converter=np.asanyarray, eq=False)
    axis_length: int = attr.ib(validator=subdtype_of(np.integer))

    axis_type: str = attr.ib(
        default="linear",
        validator=[
            subdtype_of(np.str_),
            in_(("linear", "lin", "log", "logarithmic")),
        ],
    )
    _data: np.ndarray = attr.ib(init=False, eq=False)

    @_min.validator
    @_max.validator
    def _validate_dimension_for_limits(
        self, attribute: attr.Attribute, value: np.ndarray
    ):
        if value.ndim not in (0, 1):
            raise ValueError(
                f"Wrong dimensions for '{attribute.name.replace('_', '')}'. "
                + f"Only 0d and 1d boundaries are supported."
            )

    def __attrs_post_init__(self):
        if self._max.shape != self._min.shape:
            raise ValueError("Mismatch between lower and upper limtis!")

        self._data = np.stack((self._min, self._max), axis=-1)
        if self._data.ndim == 2 and self._data.shape[0] == 1:
            self._data = np.squeeze(self._data, axis=0)

        self._min = None
        self._max = None

    def __iter__(self):
        if self.data.ndim == 1:
            yield self
        else:
            for d in self.data:
                yield self.__class__(
                    min=d[0],
                    max=d[1],
                    axis_length=self.axis_length,
                    axis_type=self.axis_type,
                    name=self.name,
                    label=self.label,
                    unit=self.unit,
                )

    def __len__(self):
        if self.data.ndim == 1:
            return self.axis_length
        else:
            return len(self.data)

    def __getitem__(self, key):
        return self.__array__()[key]

    @property
    def shape(self):
        if self.data.ndim == 1:
            return (self.axis_length,)
        else:
            return (len(self), self.axis_length)

    def _get_axis_values(self, min_, max_, N):
        if self.axis_type in ("log", "logarithmic"):
            min_, max_ = np.log10((min_, max_))
            return np.logspace(min_, max_, N)
        else:
            return np.linspace(min_, max_, N)

    def __array__(self, dtype=None):

        if self.data.ndim == 1:
            arr = self._get_axis_values(
                self._data[0], self._data[1], self.axis_length
            )

        else:
            arr = np.empty((len(self), self.axis_length))
            for i, (min_, max_) in enumerate(self._data):
                arr[i] = self._get_axis_values(min_, max_, self.axis_length)

        return arr

    def append(self, other):
        self._check_appendability(other)
        self._data = np.vstack([self.data, other.data])


@axis_attrs
class ParticleQuantity(Axis):
    _data: np.ndarray = attr.ib(converter=np.array, eq=False)
    _dtype: np.dtype = attr.ib(converter=np.dtype)
    _prt_num: np.ndarray = attr.ib(
        converter=np.array,
        validator=array_validator(dtype=np.integer),
        eq=False,
    )

    @_prt_num.validator
    def _ensure_dimensions_for_particle_array(self, attribute, value):
        if value.ndim not in (0, 1):
            raise ValueError(f"Wrong dimensions for 'particle' array")

    @_data.validator
    def _ensure_dimensions_for_data_array(self, attribute, value):
        if value.dtype == object and value.ndim not in (0, 1):
            raise ValueError(
                f"Wrong dimensions for 'data' array. "
                + "Only 0d and 1d allowed for 'data' of dtype object"
            )
        else:
            if value.ndim not in (0, 1, 2):
                raise ValueError(f"Wrong dimensions for 'data' array")

    def __attrs_post_init__(self):
        if self._prt_num.ndim == 0:
            self._prt_num = self._prt_num.reshape((1,))

        if self._data.dtype == object:
            if self._data.ndim == 0:
                self._data = self._data.reshape((1,))

        else:
            if self._data.ndim == 0:
                self._data = self._data.reshape((1, 1))

            elif self._data.ndim == 1:
                self._data = self._data.reshape((1, len(self._data)))

        if len(self._data) != len(self._prt_num):
            ValueError(
                "Length mismatch between data-array and particle numbers"
            )

    def __array__(self, dtype=None):
        if self._data.dtype == object:
            max_ = np.max(self._prt_num)
            data = np.ma.empty((len(self._data), max_), dtype=self._dtype)
            data.mask = np.ones((len(self._data), max_), dtype=np.bool)

            for i, (d, entries) in enumerate(zip(self._data, self._prt_num)):
                if isinstance(d, np.ndarray):
                    data[i, :entries] = d[:entries]
                else:
                    data[i, :entries] = d.get_data(fields=self.name)

                data.mask[i, entries:] = np.zeros(max_ - entries, dtype=np.bool)

            self._data = data

        if len(self._data) == 1:
            return np.squeeze(self._data, axis=0)

        return self._data

    def __iter__(self):
        if len(self._data) != 1:
            for d, l in zip(self._data, self._prt_num):
                yield self.__class__(
                    data=[d],
                    prt_num=[l],
                    dtype=self._dtype,
                    name=self.name,
                    label=self.label,
                    unit=self.unit,
                )
        else:
            yield self.__class__(
                data=self._data,
                prt_num=self._prt_num,
                dtype=self._dtype,
                name=self.name,
                label=self.label,
                unit=self.unit,
            )

    def __getitem__(self, key):
        return self.__array__()[key]

    @property
    def shape(self):
        if self._data.dtype == object:
            if len(self._data) == 1:
                return (np.max(self._prt_num),)
            else:
                return (len(self._data), np.max(self._prt_num))
        else:
            if len(self._data) == 1:
                return (self._data.shape[1],)

            return self._data.shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def prt_num(self):
        return self._prt_num

    def append(self, other: Union["ParticleQuantity", Any]):
        self._check_appendability(other)

        self._data = np.array(
            [d for d in self._data] + [d for d in other._data]
        )
        self._prt_num = np.concatenate((self._prt_num, other._prt_num))
