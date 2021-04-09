# -*- coding: utf-8 -*-
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import dask.array as da
import ndindex as ndx
import numpy as np


class Axis(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(
        self,
        data: da.Array,
        *,
        name: str = "unnamed",
        label: str = "unlabeled",
        unit: str = "",
        has_appendable_dim: bool = False,
    ) -> None:
        self._data = data if isinstance(data, da.Array) else da.asanyarray(data)
        self._has_appendable_dim = has_appendable_dim

        if not name.isidentifier():
            raise ValueError("Argument 'name' has to be a valid identifier")

        self._name = name
        self._label = label
        self._unit = unit

    def __len__(self) -> int:
        if self._has_appendable_dim:
            return len(self._data)
        else:
            return 1

    def __repr__(self) -> str:
        return f"Axis(name='{self.name}', label='{self.label}', unit='{self.unit}')"

    def _repr_html_(self) -> str:
        return (
            "<span>Axis</span>"
            "<span style='color: var(--jp-info-color0);'>"
            "("
            f"name='{self.name}', "
            f"label='{self.label}', "
            f"unit='{self.unit}'"
            ")"
            "</span>"
        )

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        if not new_name.isidentifier():
            raise ValueError("New name has to be a valid identifier")

        self._name = new_name

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, new_label: str) -> None:
        self._label = new_label

    @property
    def unit(self) -> str:
        return self._unit

    @unit.setter
    def unit(self, new_unit: str) -> None:
        self._unit = new_unit

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    def as_dask(self) -> da.Array:
        return self._data

    def as_numpy(self) -> np.ndarray:
        return self._data.compute()

    def __array__(self, dtype: Optional[np.dtype] = None):
        data = self._data.compute()
        return data.astype(dtype) if dtype else data

    def __getitem__(self, key: Any) -> da.Array:
        # check if appendable dimension is being reduced
        if self._has_appendable_dim:
            index = ndx.ndindex(key).expand(self._data.shape).raw
            has_appendable_dim = True if not isinstance(index[0], int) else False
        else:
            has_appendable_dim = False

        selection = self._data[key]
        return self.__class__(selection, has_appendable_dim=has_appendable_dim)

    def append(self, new_data: Union[da.Array, Any]) -> None:
        if not isinstance(new_data, da.Array):
            new_data = da.asanyarray(new_data)

        if self._has_appendable_dim:
            if new_data.ndim == (self._data.ndim - 1):
                new_data = new_data[np.newaxis]

            self._data = da.concatenate((self._data, new_data), axis=0)
        else:
            # add new dimension along which 'new_data' will be appended
            self._data = self._data[np.newaxis]
            self._has_appendable_dim = True
            new_data = new_data[np.newaxis]

            self._data = da.concatenate((self._data, new_data), axis=0)

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
        lower_limit: Union[float, int],
        upper_limit: Union[float, int],
        points: int,
        *,
        name: str = "unnamed",
        label: str = "unlabeled",
        unit: str = "",
        spacing: str = "linear",
    ) -> "Axis":
        if spacing in ("linear", "lin"):
            axis = cls._lin_axis(lower_limit, upper_limit, points)
        elif spacing in ("logarithmic", "log"):
            axis = cls._log_axis(lower_limit, upper_limit, points)
        else:
            raise ValueError(
                "Invalid axis type provided. \n"
                "Only 'lin', 'linear', 'log', and 'logarithmic' are supported"
            )

        return cls(axis, name=name, label=label, unit=unit)
