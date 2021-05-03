# -*- coding: utf-8 -*-
from textwrap import dedent
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Type
from typing import Union

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike

from .core import HasAnnotations
from .core import HasNumpyInterface


class Axis(HasAnnotations, HasNumpyInterface):
    def __init__(
        self,
        data: da.Array,
        name: str,
        label: str,
        unit: str,
    ) -> None:
        if not name.isidentifier():
            raise ValueError("Argument 'name' has to be a valid identifier")

        self._data = data
        self._name = name
        self._label = label
        self._unit = unit

    def __repr__(self) -> str:
        return f"Axis<name='{self.name}', label='{self.label}', unit='{self.unit}'>"

    def _repr_markdown_(self) -> str:
        md = f"""
        | **{type(self).__name__}** | |
        | ---: | :--- |
        | **name**  | {self.name} |
        | **label** | {self.label} |
        | **unit**  | {self.unit or "''"} |
        | **shape** | {self.shape} |
        | **dtype** | {self.dtype} |

        """
        return dedent(md)

    def __getitem__(self, key: Any) -> "Axis":
        return Axis(
            self._data[key],
            name=self._name,
            label=self._label,
            unit=self._unit,
        )

    def __hash__(self) -> int:
        return hash((self._name, self._label, self._unit))

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
    def from_array(
        cls,
        data: ArrayLike,
        *,
        name: str = "unnamed",
        label: str = "unlabeled",
        unit: str = "",
    ) -> "Axis":
        data = data if isinstance(data, da.Array) else da.asanyarray(data)
        return cls(data, name, label, unit)

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
def Axis_concatenate(
    types: Tuple[Type],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Axis:
    # repack args -> turn Axis into dask
    arrays, *rest = args
    arrays = tuple(arr.to_dask() if isinstance(arr, Axis) else arr for arr in arrays)
    args = (arrays,) + tuple(rest)
    data = da.concatenate(*args, **kwargs)
    return Axis.from_array(data)


class HasAxes:
    _axes: Tuple[Axis, ...]

    @property
    def axes(self) -> Tuple[Axis, ...]:
        return self._axes


class HasTimeAxis:
    _time: Axis

    @property
    def time(self) -> Axis:
        return self._time
