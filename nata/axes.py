# -*- coding: utf-8 -*-
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

from .types import AxisType
from .types import is_basic_indexing
from .utils.formatting import array_format
from .utils.formatting import make_identifiable


def _log_axis(min_, max_, points, dtype):
    return np.logspace(np.log10(min_), np.log10(max_), points, dtype=dtype)


def _lin_axis(min_, max_, points, dtype):
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

        self._data = data

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
        self._data = np.array(new)

    @property
    def axis_dim(self):
        return self._data.ndim

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

        if self.name != other.name:
            return False

        if self.label != other.label:
            return False

        if self.unit != other.unit:
            return False

        return True

    def append(self, other: "Axis"):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Can not append '{other}' to '{self}'")

        if not self.equivalent(other):
            raise ValueError(
                f"Mismatch in attributes between '{self}' and '{other}'"
            )

        if not self.shape and not other.shape:
            self._data = np.array([self.data, other.data], dtype=self.dtype)
        elif self.ndim == other.ndim:
            self._data = np.append(self.data, other.data, axis=0)
        else:
            self._data = np.append(self.data, other.data)


_ignored_if_data = object()


class GridAxis(Axis):
    # allowed and supported axis types are ("linear", "logarithmic", "custom")
    _axis_type_mapping = {
        "lin": "linear",
        "linear": "linear",
        "log": "logarithmic",
        "logarithmic": "logarithmic",
        "custom": "custom",
    }

    def __init__(
        self,
        min_value: Union[float, np.ndarray] = _ignored_if_data,
        max_value: Union[float, np.ndarray] = _ignored_if_data,
        grid_cells: int = _ignored_if_data,
        *,
        data: Optional[np.ndarray] = None,
        name: str = "",
        label: str = "",
        unit: str = "",
        axis_type="linear",
    ):
        if data is None:
            if any(
                arg == _ignored_if_data
                for arg in (min_value, max_value, grid_cells)
            ):
                raise ValueError(
                    "Requires 'min_value', 'max_value' and 'grid_cells' "
                    + "if 'data' is not provided!"
                )

            valid_axis_types = ("lin", "linear", "log", "logarithmic")
            if axis_type not in valid_axis_types:
                raise ValueError(
                    f"Invalid axis type. Allowed are {valid_axis_types}'."
                )

            min_value = np.asanyarray(min_value)
            max_value = np.asanyarray(max_value)

            if (
                min_value.shape != max_value.shape
                or min_value.ndim > 1
                or max_value.ndim > 1
            ):
                raise ValueError(
                    "Passed wrong parameters for min and max values!"
                )

            data = np.transpose(np.vstack((min_value, max_value)))

            super().__init__(
                data, axis_dim=1, name=name, label=label, unit=unit
            )
            self._grid_cells = grid_cells
            self._axis_type = self._axis_type_mapping[axis_type]
        else:
            data = np.asanyarray(data)

            super().__init__(
                data, axis_dim=1, name=name, label=label, unit=unit
            )

            if data.ndim == 1:
                self._grid_cells = data.shape[0]
            else:
                self._grid_cells = data.shape[1]
            self._axis_type = "custom"

    def __repr__(self) -> str:
        repr_ = f"{self.__class__.__name__}("
        repr_ += f"name='{self.name}', "
        repr_ += f"label='{self.label}', "
        repr_ += f"unit='{self.unit}', "
        repr_ += f"axis_type='{self.axis_type}', "
        repr_ += f"data={array_format(self.data)}"
        repr_ += ")"

        return repr_

    def __iter__(self) -> "GridAxis":
        if len(self) == 1:
            yield self
        else:
            for d in self._data:
                yield self.__class__(
                    data=d, name=self.name, label=self.label, unit=self.unit,
                )

    def __getitem__(self, key) -> "GridAxis":
        key = np.index_exp[key]
        if len(key) not in (1, 2):
            raise IndexError("Wrong number of indices")

        data = self.data[key]

        require_newaxis = [False, False]

        if len(key) == 1:
            if len(self) == 1:
                require_newaxis[1] = True if isinstance(key[0], int) else False
            else:
                require_newaxis[0] = True if isinstance(key[0], int) else False
        else:
            require_newaxis[0] = True if isinstance(key[0], int) else False
            require_newaxis[1] = True if isinstance(key[1], int) else False

        if any(require_newaxis):
            key_with_extension = tuple(
                np.newaxis if require else Ellipsis
                for require in require_newaxis
            )

            data = data[key_with_extension]

        return self.__class__(
            data=data, name=self.name, label=self.label, unit=self.unit,
        )

    def __array__(self, dtype=None):
        # if custom
        if self.axis_type == "custom":
            arr = self._data
        else:
            arr_dtype = dtype if dtype else self.dtype
            arr = np.zeros((len(self._data), self._grid_cells), dtype=arr_dtype)

            if self.axis_type == "linear":
                for i, (min_, max_) in enumerate(self._data):
                    arr[i, :] = _lin_axis(
                        min_, max_, self._grid_cells, arr_dtype
                    )

            if self.axis_type == "logarithmic":
                for i, (min_, max_) in enumerate(self._data):
                    arr[i, :] = _log_axis(
                        min_, max_, self._grid_cells, arr_dtype
                    )
        if len(self) == 1:
            return np.squeeze(arr, axis=0)
        else:
            return arr

    @property
    def data(self):
        return self.__array__()

    @data.setter
    def data(
        self, value: Union[np.ndarray, Sequence[Union[int, float]]]
    ) -> None:
        raise NotImplementedError  # TODO: consider the best way to store data

    @property
    def grid_cells(self):
        return self._grid_cells

    @property
    def axis_type(self):
        return self._axis_type

    @axis_type.setter
    def axis_type(self, value: str) -> None:
        if not isinstance(value, str):
            value = str(value)

        if value in self._axis_type_mapping:
            self._axis_type = self._axis_type_mapping[value]
        else:
            allowed = tuple(t for t in self._axis_type_mapping if t != "custom")
            raise ValueError(
                "Invalid axis type provided. Following axis types are "
                + f"{allowed}."
            )

    @property
    def shape(self):
        if len(self._data) == 1:
            return (self._grid_cells,)
        else:
            return (len(self._data), self._grid_cells)

    @property
    def ndim(self):
        if len(self._data) == 1:
            return 1
        else:
            return 2

    def equivalent(self, other: Union[Any, "GridAxis"]) -> bool:
        if not super().equivalent(other):
            return False

        if self.axis_type[:3] != other.axis_type[:3]:
            return False

        return True

    def append(self, other: "GridAxis"):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Can not append '{other}' to '{self}'")

        if not self.equivalent(other):
            raise ValueError(
                f"Mismatch in attributes between '{self}' and '{other}'"
            )

        self._data = np.append(self._data, other._data, axis=0)
