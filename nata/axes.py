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


def _log_axis(min_, max_, points):
    return np.logspace(np.log10(min_), np.log10(max_), points)


def _lin_axis(min_, max_, points):
    return np.linspace(min_, max_, points)


class Axis:
    def __init__(
        self,
        data: np.ndarray,
        *,
        axis_dim: int = 0,
        name: str = "unnamed",
        label: str = "",
        unit: str = "",
    ):
        if not isinstance(data, np.ndarray):
            data = np.asanyarray(data)

        self._data = data
        self._axis_dim = axis_dim

        name = make_identifiable(name)
        self._name = name if name else "unnamed"
        self._label = label
        self._unit = unit

    def __repr__(self) -> str:
        repr_ = f"{self.__class__.__name__}("
        repr_ += f"name='{self.name}', "
        repr_ += f"label='{self.label}', "
        repr_ += f"unit='{self.unit}', "
        repr_ += f"axis_dim={self.axis_dim}"
        repr_ += ")"

        return repr_

    def __len__(self) -> int:
        if self.shape == ():
            return 1
        else:
            return len(self._data)

    def __iter__(self) -> "Axis":
        if self.shape == ():
            data = self.data[np.newaxis]
        else:
            data = self.data

        for d in data:
            yield self.__class__(
                d, name=self.name, label=self.label, unit=self.unit,
            )

    def __getitem__(
        self, key: Union[int, slice, Tuple[Union[int, slice]]]
    ) -> "Axis":
        if not is_basic_indexing(key):
            raise IndexError("Only basic indexing is supported!")

        return self.__class__(
            self.data[key], name=self.name, label=self.label, unit=self.unit,
        )

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        if dtype:
            return self._data.astype(dtype)
        else:
            return self._data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: Union[np.ndarray, Any]) -> None:
        new = np.broadcast_to(value, self.shape, subok=True)
        self._data = np.array(new, subok=True)

    @property
    def axis_dim(self):
        return self._axis_dim

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
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        parsed_value = make_identifiable(str(value))
        if not parsed_value:
            raise ValueError("Invalid name provided!")
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
    _axis_type_mapping = {
        "lin",
        "linear",
        "log",
        "logarithmic",
        "custom",
    }

    def __init__(
        self,
        data: np.ndarray,
        *,
        axis_dim: int = 1,
        axis_type: str = "linear",
        name: str = "unnamed",
        label: str = "",
        unit: str = "",
    ) -> None:
        super().__init__(
            data, axis_dim=axis_dim, name=name, label=label, unit=unit
        )

        if axis_type not in self._axis_type_mapping:
            raise ValueError(
                f"'{axis_type}' is not supported for axis_type! "
                + f"It has to by one of {self._axis_type_mapping}"
            )
        self._axis_type = axis_type

    def __iter__(self) -> "GridAxis":
        data = self.data if self.ndim else self.data[np.newaxis]

        for d in data:
            yield self.__class__(
                d,
                axis_dim=self.axis_dim,
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

        return self.__class__(
            self.data[key],
            axis_dim=self.axis_dim,
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
        repr_ += f"axis_dim={self.axis_dim}, "
        repr_ += f"axis_type={self.axis_type}"
        repr_ += ")"

        return repr_

    @property
    def axis_type(self) -> str:
        return self._axis_type

    @axis_type.setter
    def axis_type(self, value: str) -> None:
        value = str(value)
        if value not in self._axis_type_mapping:
            raise ValueError(
                f"'{value}' is not supported for axis_type! "
                + f"It has to by one of {self._axis_type_mapping}"
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
            axis = _lin_axis(min_value, max_value, cells)
        elif axis_type in ("log", "logarithmic"):
            axis = _log_axis(min_value, max_value, cells)
        else:
            raise ValueError(
                "Invalid axis type provided. "
                + "Only 'lin', 'linear', 'log', and 'logarithmic' "
                + "are supported!"
            )
        axis = cls(axis, name=name, label=label, unit=unit)
        axis._axis_type = axis_type
        return axis

    def equivalent(self, other: Union[Any, GridAxisType]) -> bool:
        if not super().equivalent(other):
            return False

        if self.axis_type != other.axis_type:
            return False

        return True
