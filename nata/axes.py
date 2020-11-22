# -*- coding: utf-8 -*-
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np

import dask.array as da
import ndindex as ndx

from .types import AxisType
from .utils.exceptions import DimensionError
from .utils.formatting import make_identifiable


class Axis(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(
        self,
        data: da.core.Array,
        *,
        axis_dim: Optional[int] = None,
        name: str = "unnamed",
        label: str = "unlabeled",
        unit: str = "",
    ):
        self._data = data if isinstance(data, da.core.Array) else da.asanyarray(data)
        self._axis_dim = self._data.ndim if axis_dim is None else axis_dim

        if (self._axis_dim < 0) or (self._data.ndim < self._axis_dim):
            raise DimensionError("Mismatch between data and axis dimensionality!")

        cleaned_name = make_identifiable(name)
        self._name = cleaned_name if cleaned_name else "unnamed"
        self._label = label
        self._unit = unit

    def __repr__(self) -> str:
        return f"{type(self).__name__}<name={self.name}, axis_dim={self.axis_dim}>"

    def __len__(self) -> int:
        if self._data.ndim > self._axis_dim:
            return self._data.shape[0]
        else:
            return 1

    def __iter__(self) -> "Axis":
        if len(self) == 1:
            yield self
        else:
            for d in self._data:
                self.__class__(
                    d[np.newaxis],
                    axis_dim=self.axis_dim,
                    name=self.name,
                    label=self.label,
                    unit=self.unit,
                )

    def __getitem__(self, key: Any) -> "Axis":
        if not self.shape:
            raise IndexError("Can not index 0-dimensional axis")

        index = ndx.ndindex(key).expand(self.shape).raw
        data = self._data[index]
        axis_dim = self._axis_dim

        # reduction of axis dimensionality
        # - dimension associated with axis are the most right dimensions in `.shape`
        # - `ind_associated_to_dim >= 0` following from `__init__`
        ind_associated_to_dim = self._data.ndim - self._axis_dim
        count_int = sum(isinstance(ind, int) for ind in index[ind_associated_to_dim:])
        axis_dim = (axis_dim - count_int) if (axis_dim - count_int) > 0 else 0

        return self.__class__(
            data, axis_dim=axis_dim, name=self.name, label=self.label, unit=self.unit,
        )

    def __setitem__(
        self, key: Union[int, slice, Tuple[Union[int, slice]]], value: Any
    ) -> None:
        self.data[key] = value

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        return self.as_numpy().astype(dtype) if dtype else self.as_numpy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> Optional["Axis"]:
        inputs = tuple(self._data if in_ is self else in_ for in_ in inputs)

        if "out" in kwargs:
            kwargs["out"] = tuple(
                self._data if in_ is self else in_ for in_ in kwargs["out"]
            )

        data = self._data.__array_ufunc__(ufunc, method, *inputs, **kwargs)

        if data is NotImplemented:
            raise NotImplementedError(
                f"ufunc '{ufunc}' "
                + f"for {method=}, "
                + f"{inputs=}, "
                + f"and {kwargs=} not implemented!"
            )
        elif data is None:
            # in-place
            self._data = kwargs["out"][0]
            return self
        else:
            return self.__class__(
                data,
                axis_dim=self.axis_dim,
                name=self.name,
                label=self.label,
                unit=self.unit,
            )

    def __array_function__(
        self,
        func: Callable,
        types: Tuple[Type[Any], ...],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        # repack arguments
        types = tuple(type(self._data) if t is type(self) else t for t in types)
        args = tuple(self._data if arg is self else arg for arg in args)

        data = self._data.__array_function__(func, types, args, kwargs)
        return self.__class__(
            data,
            axis_dim=self.axis_dim,
            name=self.name,
            label=self.label,
            unit=self.unit,
        )

    @property
    def data(self) -> np.ndarray:
        return np.asanyarray(self)

    @data.setter
    def data(self, value: Union[np.ndarray, Any]) -> None:
        new = np.broadcast_to(value, self.shape, subok=True)
        self._data = da.asanyarray(new)

    @property
    def axis_dim(self) -> int:
        return self._axis_dim

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value) -> None:
        parsed_value = make_identifiable(str(value))
        if not parsed_value:
            raise ValueError("Invalid name provided! Name requires to be identifiable!")
        self._name = parsed_value

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value) -> None:
        value = str(value)
        self._label = value

    @property
    def unit(self) -> str:
        return self._unit

    @unit.setter
    def unit(self, value) -> None:
        value = str(value)
        self._unit = value

    @staticmethod
    def _log_axis(
        min_: Union[float, int], max_: Union[float, int], points: int
    ) -> da.core.Array:
        """Generates logarithmically spaced array."""
        min_ = np.log10(min_)
        max_ = np.log10(max_)
        return 10.0 ** da.linspace(min_, max_, points)

    @staticmethod
    def _lin_axis(
        min_: Union[float, int], max_: Union[float, int], points: int
    ) -> da.core.Array:
        """Generates linearly spaced array."""
        return da.linspace(min_, max_, points)

    @classmethod
    def from_limits(
        cls,
        min_value: Union[int, float],
        max_value: Union[int, float],
        cells: int,
        *,
        name: str = "unnamed",
        label: str = "unlabeled",
        unit: str = "",
        axis_type: str = "linear",
    ) -> "Axis":
        if axis_type in ("linear", "lin"):
            axis = cls._lin_axis(min_value, max_value, cells)
        elif axis_type in ("logarithmic", "log"):
            axis = cls._log_axis(min_value, max_value, cells)
        else:
            raise ValueError(
                "Invalid axis type provided. "
                + "Only 'lin', 'linear', 'log', and 'logarithmic' are supported!"
            )

        return cls(axis, axis_dim=1, name=name, label=label, unit=unit)

    def is_equiv_to(self, other: Union[Any, AxisType]) -> bool:
        return isinstance(other, self.__class__) and all(
            getattr(self, prop) == getattr(other, prop)
            for prop in ("axis_dim", "name", "label", "unit")
        )

    def append(self, other: "Axis") -> "Axis":
        if not isinstance(other, self.__class__):
            raise TypeError(f"Can not append '{other}' to '{self}'")

        if not self.is_equiv_to(other):
            raise ValueError(f"Mismatch in attributes between '{self}' and '{other}'")

        # transform self array to ensure one extra dimension exist for stacking
        if self._data.ndim == self._axis_dim:
            self_data = self._data[np.newaxis]
        else:
            self_data = self._data

        # transform other array to ensure one extra dimension exist for stacking
        if other.ndim == other.axis_dim:
            other_data = other.as_dask(squeeze=False)[np.newaxis]
        else:
            other_data = other.as_dask(squeeze=False)

        self._data = da.concatenate((self_data, other_data), axis=0)

    def as_numpy(self, squeeze: bool = False) -> np.ndarray:
        if squeeze:
            return np.asanyarray(da.squeeze(self._data).compute())
        else:
            return np.asanyarray(self._data.compute())

    def as_dask(self, squeeze: bool = False) -> da.core.Array:
        if squeeze:
            return da.asanyarray(da.squeeze(self._data))
        else:
            return self._data
