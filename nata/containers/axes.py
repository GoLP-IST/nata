from copy import copy
from typing import Dict
from typing import Any
from typing import Union
from typing import Tuple

import attr
from attr.validators import instance_of
from attr.validators import optional
from attr.validators import deep_iterable
from attr.validators import deep_mapping
from attr.validators import in_
import numpy as np

from nata.backends.grid import BaseGrid

_incomparable = {"order": False, "eq": False}


@attr.s(init=False, slots=True)
class Axis:
    _parent = attr.ib(**_incomparable)
    _dtype = attr.ib(validator=instance_of((type, np.dtype)))
    _mapping: Dict[int, Any] = attr.ib(**_incomparable)
    label: str = attr.ib(validator=instance_of(str))
    unit: str = attr.ib(validator=instance_of(str))

    def __init__(self, parent, key, value, label, unit, dtype=int):
        self._parent = parent
        self._dtype = dtype
        self._mapping = {key: value}
        self.label = label
        self.unit = unit
        attr.validate(self)

    def update(self, other):
        """Takes Axis `other` and consumes its storage"""
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Requires other object to be of type `{self.__class__}`"
            )
        if self != other:
            raise ValueError("Only equal axis can be updated")

        self._mapping.update(other._mapping)

    def __len__(self) -> int:
        """Entries inside axis"""
        return len(self._mapping)

    def update_mapping(self, to_keep):
        """Updating mappings in Axis"""
        self._mapping = {k: v for k, v in self._mapping.items() if k in to_keep}

    def _update_parent(self, to_keep):
        """Updating iteration stored in parent

        Method to create a shallow copy of `._parent` object and modifies keys
        inside the new parent object.
        """
        # insures that `to_keep is really an iteratorable object`
        # https://docs.python.org/3.8/library/collections.abc.html#collections.abc.Iterable
        try:
            iter(to_keep)
        except TypeError:
            raise TypeError("Requires an iterator")

        new_parent = copy(self._parent)
        new_parent.iterations_to_keep(to_keep)
        return new_parent

    def asarray(self, with_keys: bool = False) -> np.ndarray:
        array_dtype = np.dtype([("key", int), ("value", self._dtype)])
        arr = np.fromiter(self._mapping.items(), dtype=array_dtype)
        if with_keys:
            return arr["key"], arr["value"]
        return arr["value"]

    def keys(self):
        return self._mapping.keys()

    def values(self):
        return self._mapping.values()

    def items(self):
        return self._mapping.items()

    def __iter__(self):
        for k in self.values():
            yield k


@attr.s(init=False, slots=True)
class IterationAxis(Axis):
    _mapping: Dict[int, int] = attr.ib(
        **_incomparable,
        repr=False,
        validator=deep_mapping(
            key_validator=optional(instance_of(int)),
            value_validator=instance_of(int),
            mapping_validator=instance_of(dict),
        ),
    )

    def __init__(self, parent, key, value, label="iteration", unit=""):
        if not isinstance(value, int):
            raise TypeError("Requires `value` to be of type `int`")

        super().__init__(
            parent, key, value, label=label, unit=unit, dtype=np.int
        )

    def __getitem__(self, key: Union[int, float, slice]):
        if not isinstance(key, (int, float, slice)):
            raise TypeError(
                f"Only `int`, `float` and `slice` are allowed for `{self}`"
            )

        values = self.asarray(with_keys=False)

        # determin which keys to keep (neigherst neighbor or range)
        if isinstance(key, (int, float)):
            values_to_keep = values[np.argmin(np.abs(values - key))]
        else:
            start = key.start if key.start else values.min()
            stop = key.stop if key.stop else values.max()
            values_to_keep = values[(start <= values) * (values <= stop)]

        if isinstance(values_to_keep, np.ndarray):
            values_to_keep = set(values_to_keep)
        else:
            values_to_keep = set((values_to_keep,))

        return super()._update_parent(values_to_keep)


@attr.s(init=False, slots=True)
class TimeAxis(Axis):
    _mapping: Dict[int, float] = attr.ib(
        **_incomparable,
        repr=False,
        validator=deep_mapping(
            key_validator=instance_of(int),
            value_validator=instance_of(float),
            mapping_validator=instance_of(dict),
        ),
    )

    def __init__(self, parent, key, value, label="time", unit=""):
        if not isinstance(value, float):
            raise TypeError("Requires `value` to be of type `float`")

        super().__init__(
            parent, key, value, label=label, unit=unit, dtype=np.float
        )

    def __getitem__(self, key: Union[int, float, slice]):
        if not isinstance(key, (int, float, slice)):
            raise TypeError(
                f"Only `int`, `float` and `slice` are allowed for `{self}`"
            )

        values = self.asarray(with_keys=False)

        # determin which keys to keep (neigherst neighbor or range)
        if isinstance(key, (int, float)):
            values_to_keep = values[np.argmin(np.abs(values - key))]
        else:
            start = key.start if key.start else values.min()
            stop = key.stop if key.stop else values.max()
            values_to_keep = values[(start <= values) * (values <= stop)]

        if isinstance(values_to_keep, np.ndarray):
            values_to_keep = set(values_to_keep)
        else:
            values_to_keep = set((values_to_keep,))

        return super()._update_parent(values_to_keep)


@attr.s(init=False, slots=True)
class GridAxis(Axis):
    name: str = attr.ib(validator=instance_of(str))
    length: int = attr.ib(validator=instance_of(int))
    axis_type: str = attr.ib(validator=in_(["linear", "logarithmic"]))
    _mapping: Dict[int, Tuple[float, float]] = attr.ib(
        **_incomparable,
        repr=False,
        validator=deep_mapping(
            key_validator=instance_of(int),
            value_validator=instance_of(tuple),
            mapping_validator=instance_of(dict),
        ),
    )

    def __init__(
        self,
        parent,
        key,
        value,
        name,
        length,
        label="",
        unit="",
        axis_type="linear",
    ):
        if not isinstance(value, (tuple, list, np.ndarray)):
            raise TypeError("Requires `value` to be array-like")

        if not len(value) == 2:
            raise ValueError("`value` has to be a tuple with two entries")

        value = tuple(sorted(value))
        self.axis_type = axis_type
        self.name = name
        self.length = length

        super().__init__(
            parent, key, value, label=label, unit=unit, dtype=float
        )

    def asarray(self, with_keys: bool = False) -> np.ndarray:
        array_dtype = np.dtype([("key", int), ("values", float, (2,))])

        arr = np.fromiter(self._mapping.items(), dtype=array_dtype)
        if with_keys:
            return arr["key"], arr["values"]
        return arr["values"]

    @property
    def axis_values(self):
        data = np.zeros((len(self), self.length), dtype=self._dtype)

        if self.axis_type == "logarithmic":
            for i, (min_, max_) in zip(
                range(len(self)), self._mapping.values()
            ):
                data[i] = np.logspace(
                    np.log10(min_), np.log10(max_), self.length
                )
        else:
            for i, (min_, max_) in zip(
                range(len(self)), self._mapping.values()
            ):
                data[i] = np.linspace(min_, max_, self.length)

        return data

    @property
    def min(self):
        return np.fromiter(
            [k[0] for k in self._mapping.values()], dtype=self._dtype
        )

    @min.setter
    def min(self, value):
        if len(self._mapping) != len(value):
            raise ValueError("wrong number of elements to set min")

        for key, val in zip(self._mapping.keys(), value):
            self._mapping[key] = (val, self._mapping[key][1])

    @property
    def max(self):
        return np.fromiter(
            [k[1] for k in self._mapping.values()], dtype=self._dtype
        )

    @max.setter
    def max(self, value):
        if len(self._mapping) != len(value):
            raise ValueError("wrong number of elements to set max")

        for key, val in zip(self._mapping.keys(), value):
            self._mapping[key] = (self._mapping[key][0], val)


@attr.s(init=False, slots=True)
class DataStock:
    _mapping: Dict[int, Union[np.ndarray, BaseGrid]] = attr.ib(
        repr=False,
        **_incomparable,
        validator=deep_mapping(
            key_validator=instance_of(int),
            value_validator=instance_of((BaseGrid, np.ndarray)),
            mapping_validator=instance_of(dict),
        ),
    )
    shape: Tuple[int] = attr.ib(
        validator=deep_iterable(instance_of(int), instance_of(tuple))
    )
    dim: int = attr.ib(validator=instance_of(int))
    dtype: np.dtype = attr.ib(validator=instance_of((type, np.dtype)))

    def __init__(
        self, key: int, value: BaseGrid, shape: Tuple, dtype: np.dtype
    ):
        if isinstance(value, np.ndarray):
            value = value.reshape((1,) + shape)
        self._mapping = {key: value}
        self.shape = shape
        self.dim = len(shape)
        self.dtype = dtype
        attr.validate(self)

    def __getitem__(self, index: Union[slice, int]) -> np.ndarray:
        if not isinstance(index, (slice, int)):
            raise KeyError(f"invalid index for {self}")

        if isinstance(index, int):
            if not isinstance(self._mapping[index], np.ndarray):
                data = self._mapping[index].dataset
                self._mapping[index] = data.astype(self.dtype).reshape(
                    (1,) + self.shape
                )
            return self._mapping[index]

        # only non-specifc index are allowed as slices for now
        if any(s is not None for s in (index.start, index.stop, index.step)):
            raise NotImplementedError("Currently slices are not implemented")

        return_shape = (len(self._mapping),) + self.shape
        full_array = np.zeros(return_shape, dtype=self.dtype)
        for i, val in enumerate(self._mapping.values()):
            if isinstance(val, np.ndarray):
                full_array[i] = val
            else:
                full_array[i] = val.dataset
        return full_array

    def __setitem__(self, index, value) -> None:
        if not isinstance(index, (slice, int)):
            raise KeyError(f"Invalid index for {self}")

        if not isinstance(value, np.ndarray):
            raise NotImplementedError(
                "Only assignment with numpy arrays are valid"
            )

        if isinstance(index, int):
            required_shape = (1,) + self.shape
            if value.shape != self.shape and value.shape != required_shape:
                raise ValueError(
                    f"Can not broadcast {value.shape} to {required_shape}"
                )

            value = value.reshape(required_shape)
            self._mapping[index] = value.astype(self.dtype, copy=False)

        else:
            if any(
                s is not None for s in (index.start, index.stop, index.step)
            ):
                raise NotImplementedError(
                    "Currently slices are not implemented"
                )

            total_shape = (len(self._mapping),) + self.shape

            if (value.shape != total_shape) and (value.shape != self.shape):
                raise ValueError(
                    f"Can not broadcast {value.shape} to "
                    + f"{total_shape} or {self.shape}"
                )

            if value.shape == total_shape:
                for key, arr in zip(self._mapping.keys(), value):
                    self._mapping[key] = arr
            else:
                for key in self._mapping.keys():
                    self._mapping[key] = value

    def update(self, other):
        if self != other:
            raise TypeError(f"Can not append {other} to {self}")

        self._mapping.update(other._mapping)
