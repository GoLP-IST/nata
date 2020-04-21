# -*- coding: utf-8 -*-
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from .types import AxisType
from .types import GridAxisType
from .types import is_basic_indexing
from .utils.formatting import make_identifiable


def _log_axis(
    min_: Union[float, np.ndarray], max_: Union[float, np.ndarray], points: int
) -> np.ndarray:
    """Generates logarithmically spaced axis/array.

    Returns always an array with the shape (points,) + np.shape(min/max) and
    a floating dtype.
    """
    if np.issubdtype(
        # min_
        type(min_) if not isinstance(min_, np.ndarray) else min_.dtype,
        np.floating,
    ) or np.issubdtype(
        # max_
        type(max_) if not isinstance(max_, np.ndarray) else max_.dtype,
        np.floating,
    ):
        dtype = None
    else:
        dtype = float

    return np.logspace(np.log10(min_), np.log10(max_), points, dtype=dtype)


def _lin_axis(
    min_: Union[float, np.ndarray], max_: Union[float, np.ndarray], points: int
) -> np.ndarray:
    """Generates linearly spaced axis/array.

    Returns always an array with the shape (points,) + np.shape(min/max) and
    a floating dtype.
    """
    if np.issubdtype(
        # min_
        type(min_) if not isinstance(min_, np.ndarray) else min_.dtype,
        np.floating,
    ) or np.issubdtype(
        # max_
        type(max_) if not isinstance(max_, np.ndarray) else max_.dtype,
        np.floating,
    ):
        dtype = None
    else:
        dtype = float

    return np.linspace(min_, max_, points, dtype=dtype)


class Axis:
    def __init__(
        self,
        data: np.ndarray,
        *,
        name: str = "unnamed",
        label: str = "",
        unit: str = "",
    ):
        if not isinstance(data, np.ndarray):
            data = np.asanyarray(data)

        self._data = data if data.ndim > 0 else data[np.newaxis]

        name = make_identifiable(name)
        self._name = name if name else "unnamed"
        self._label = label
        self._unit = unit

    def __repr__(self) -> str:
        repr_ = f"{self.__class__.__name__}("
        repr_ += f"name='{self.name}', "
        repr_ += f"label='{self.label}', "
        repr_ += f"unit='{self.unit}', "
        repr_ += f"axis_dim={self.axis_dim}, "
        repr_ += f"len={len(self)}"
        repr_ += ")"

        return repr_

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> "Axis":
        for d in self._data:
            yield self.__class__(
                d[np.newaxis], name=self.name, label=self.label, unit=self.unit,
            )

    def __getitem__(
        self, key: Union[int, slice, Tuple[Union[int, slice]]]
    ) -> "Axis":
        if not is_basic_indexing(key):
            raise IndexError("Only basic indexing is supported!")

        key = np.index_exp[key]
        requires_new_axis = False

        # > determine if axis extension is required
        # 1st index (temporal slicing) not hidden if ndim == axis_dim + 1
        # or alternatively -> check len of the axis -> number of temporal slices
        if len(self) != 1:
            # revert dimensionality reduction
            if isinstance(key[0], int):
                requires_new_axis = True
        else:
            requires_new_axis = True

        data = self.data[key]

        if requires_new_axis:
            data = data[np.newaxis]

        return self.__class__(
            data, name=self.name, label=self.label, unit=self.unit,
        )

    def __setitem__(
        self, key: Union[int, slice, Tuple[Union[int, slice]]], value: Any
    ) -> None:
        self.data[key] = value

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        data = self._data.astype(dtype) if dtype else self._data
        return np.squeeze(data, axis=0) if len(self) == 1 else data

    @property
    def data(self) -> np.ndarray:
        return np.asanyarray(self)

    @data.setter
    def data(self, value: Union[np.ndarray, Any]) -> None:
        new = np.broadcast_to(value, self.shape, subok=True)
        if len(self) == 1:
            self._data = np.array(new, subok=True)[np.newaxis]
        else:
            self._data = np.array(new, subok=True)

    @property
    def axis_dim(self):
        return self._data.ndim - 1

    @property
    def shape(self):
        return self._data.shape[1:] if len(self) == 1 else self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return (self._data.ndim - 1) if len(self) == 1 else self._data.ndim

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        parsed_value = make_identifiable(str(value))
        if not parsed_value:
            raise ValueError(
                "Invalid name provided! Has to be able to be valid code"
            )
        self._name = parsed_value

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        value = str(value)
        self._label = value

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        value = str(value)
        self._unit = value

    def equivalent(self, other: Union[Any, AxisType]) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if self.axis_dim != other.axis_dim:
            return False

        if self.name != other.name:
            return False

        if self.label != other.label:
            return False

        if self.unit != other.unit:
            return False

        return True

    def append(self, other: "Axis") -> "Axis":
        if not isinstance(other, self.__class__):
            raise TypeError(f"Can not append '{other}' to '{self}'")

        if not self.equivalent(other):
            raise ValueError(
                f"Mismatch in attributes between '{self}' and '{other}'"
            )

        selfdata = (
            self.data[np.newaxis] if self.ndim == self.axis_dim else self.data
        )

        otherdata = (
            other.data[np.newaxis]
            if other.ndim == other.axis_dim
            else other.data
        )

        self._data = np.append(selfdata, otherdata, axis=0)


_ignored_if_data = object()


class GridAxis(Axis):
    _supported_axis_types: Tuple[str, ...] = (
        "lin",
        "linear",
        "log",
        "logarithmic",
        "custom",
    )

    def __init__(
        self,
        data: np.ndarray,
        *,
        axis_type: str = "linear",
        name: str = "unnamed",
        label: str = "",
        unit: str = "",
    ) -> None:
        if axis_type not in self._supported_axis_types:
            raise ValueError(
                f"'{axis_type}' is not supported for axis_type! "
                + f"It has to by one of {self._supported_axis_types}"
            )

        super().__init__(data, name=name, label=label, unit=unit)
        self._axis_type = axis_type

    def __iter__(self) -> "GridAxis":
        for d in self._data:
            yield self.__class__(
                d[np.newaxis],
                name=self.name,
                label=self.label,
                unit=self.unit,
                axis_type=self.axis_type,
            )

    def __getitem__(
        self, key: Union[int, slice, Tuple[Union[int, slice]]]
    ) -> "GridAxis":
        if not is_basic_indexing(key):
            raise IndexError("Only basic indexing is supported!")

        key = np.index_exp[key]
        requires_new_axis = False

        # first index corresponds to temporal slicing if ndim == axis_dim + 1
        # or alternatively -> check len of the axis -> number of temporal slices
        if len(self) != 1:
            # revert dimensionality reduction
            if isinstance(key[0], int):
                requires_new_axis = True
        else:
            requires_new_axis = True

        return self.__class__(
            self.data[key][np.newaxis] if requires_new_axis else self.data[key],
            name=self.name,
            label=self.label,
            unit=self.unit,
            axis_type=self.axis_type,
        )

    def __repr__(self) -> str:
        repr_ = f"{self.__class__.__name__}("
        repr_ += f"name='{self.name}', "
        repr_ += f"label='{self.label}', "
        repr_ += f"unit='{self.unit}', "
        repr_ += f"axis_type={self.axis_type}, "
        repr_ += f"axis_dim={self.axis_dim}, "
        repr_ += f"len={len(self)}"
        repr_ += ")"

        return repr_

    @property
    def axis_type(self) -> str:
        return self._axis_type

    @axis_type.setter
    def axis_type(self, value: str) -> None:
        value = str(value)
        if value not in self._supported_axis_types:
            raise ValueError(
                f"'{value}' is not supported for axis_type! "
                + f"It has to by one of {self._supported_axis_types}"
            )
        self._axis_type = value

    @classmethod
    def from_limits(
        cls,
        min_value: Union[np.ndarray, int, float],
        max_value: Union[np.ndarray, int, float],
        cells: int,
        *,
        axis_type: str = "linear",
        name: str = "unnamed",
        label: str = "",
        unit: str = "",
    ) -> "GridAxis":
        if axis_type in ("lin", "linear"):
            axis: np.ndarray = _lin_axis(min_value, max_value, cells)
        elif axis_type in ("log", "logarithmic"):
            axis: np.ndarray = _log_axis(min_value, max_value, cells)
        else:
            raise ValueError(
                "Invalid axis type provided. "
                + "Only 'lin', 'linear', 'log', and 'logarithmic' "
                + "are supported!"
            )

        if axis.ndim == 1:
            axis = axis[np.newaxis]

        axis = cls(axis, name=name, label=label, unit=unit)
        axis._axis_type = axis_type
        return axis

    def equivalent(self, other: Union[Any, GridAxisType]) -> bool:
        if not super().equivalent(other):
            return False

        if self.axis_type != other.axis_type:
            return False

        return True
