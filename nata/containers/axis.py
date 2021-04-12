# -*- coding: utf-8 -*-
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import dask.array as da
import ndindex as ndx
import numpy as np

from .formatting import Table


class Axis(np.lib.mixins.NDArrayOperatorsMixin):
    _handled_array_function = {}
    _handled_array_ufunc = {}

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
        html = Table(
            f"{type(self).__name__}",
            {
                "name": self.name,
                "label": self.label,
                "unit": self.unit or "''",
                "ndim": self.ndim,
                "shape": self.shape,
                "dtype": self.dtype,
            },
            fold_closed=False,
        ).render_as_html()
        return html

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

    @classmethod
    def implements(
        cls,
        numpy_function: Callable,
        function_type: str = "array_function",
    ):
        def decorator(func):
            if function_type == "array_function":
                cls._handled_array_function[numpy_function] = func
            elif function_type == "array_ufunc":
                cls._handled_array_ufunc[numpy_function] = func
            else:
                raise ValueError("Invalid 'function_type' provided")

            return func

        return decorator

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    ) -> "Axis":
        if ufunc in self._handled_array_ufunc:
            return self._handled_array_ufunc[ufunc](method, *inputs, **kwargs)

        # Takes inputs and replaces instances of 'self' by '_data'
        inputs = tuple(self._data if obj is self else obj for obj in inputs)

        if "out" in kwargs:
            kwargs["out"] = tuple(
                self._data if obj is self else obj for obj in kwargs["out"]
            )

        data = self._data.__array_ufunc__(ufunc, method, *inputs, **kwargs)

        if data is NotImplemented:
            raise NotImplementedError(
                f"{ufunc=} for {method=}, {inputs=}, and {kwargs=} not implemented"
            )
        elif data is None:
            # in-place
            self._data = kwargs["out"][0]
            return self
        else:
            return self.__class__(
                data,
                name=self.name,
                label=self.label,
                unit=self.unit,
                has_appendable_dim=self._has_appendable_dim,
            )

    def __array_function__(
        self,
        function: Callable,
        types: Tuple[Type[Any], ...],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        if function in self._handled_array_function:
            return self._handled_array_function[function](*args, **kwargs)

        repacked_types = tuple(
            type_ if not issubclass(type_, Axis) else da.Array for type_ in types
        )
        repacked_args = tuple(
            arg if not isinstance(arg, Axis) else arg.as_dask() for arg in args
        )

        data = self._data.__array_function__(
            function,
            repacked_types,
            repacked_args,
            kwargs,
        )

        # TODO: Are name, label, and unit required to be passed here or should we have
        #       default option
        return self.__class__(data, has_appendable_dim=self._has_appendable_dim)

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
        if isinstance(new_data, Axis):
            new_data = new_data.as_dask()

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


@Axis.implements(np.concatenate)
def concatenate(arrays, *args, **kwargs):
    arrays = tuple(
        array if not isinstance(array, Axis) else array.as_dask() for array in arrays
    )
    return Axis(np.concatenate(arrays, *args, **kwargs))
