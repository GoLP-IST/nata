# -*- coding: utf-8 -*-
from textwrap import dedent
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import dask.array as da
from numpy.typing import ArrayLike

from .axis import Axis
from .axis import HasTimeAxis
from .core import HasAnnotations
from .core import HasNumpyInterface
from .core import HasPluginSystem


class HasParticleCount:
    _num: Union[int, Tuple[int, ...]]

    @property
    def num(self) -> Union[int, Tuple[int, ...]]:
        return self._num


class Quantity(
    HasNumpyInterface, HasAnnotations, HasPluginSystem, HasTimeAxis, HasParticleCount
):
    def __init__(
        self, data: da.Array, time: Axis, name: str, label: str, unit: str
    ) -> None:
        if data.ndim != 0:
            raise ValueError("only 0d data is supported")

        if data.dtype.fields:
            raise ValueError("only unstructured data types supported")

        self._name = name
        self._label = label
        self._unit = unit

        self._num = 1
        self._time = time

        self._data = data

    def __getitem__(self, key: Any) -> Union["Quantity", "QuantityArray"]:
        raise NotImplementedError

    def __hash__(self) -> int:
        # general naming
        key = (self.name, self.label, self.unit, self.shape)

        # time specific
        key += (self.time.name, self.time.label, self.time.unit)

        # number of particles
        key += (self.num,)

        return hash(key)

    def __repr__(self) -> str:
        repr_ = (
            f"{type(self).__name__}<"
            f"{self.to_numpy()}, "
            f"dtype={self.dtype}, "
            f"time={self.time.to_numpy()}"
            ">"
        )
        return repr_

    def _repr_markdown_(self) -> str:
        md = f"""
        | **{type(self).__name__}** | |
        | ---: | :--- |
        | **name**  | {self.name} |
        | **label** | {self.label} |
        | **unit**  | {self.unit or "''"} |
        | **dtype** | {self.dtype} |
        | **time**  | {self.time.to_numpy()} |

        """
        return dedent(md)

    @classmethod
    def from_array(
        cls,
        data: ArrayLike,
        *,
        name: str = "unnamed",
        label: str = "unlabeled",
        unit: str = "",
        time: Optional[Union[Axis, int, float]] = None,
    ) -> "Quantity":
        data = data if isinstance(data, da.Array) else da.asanyarray(data)

        if time is None:
            time = Axis.from_array(0.0, name="time", label="time")
        else:
            if not isinstance(time, Axis):
                time = Axis.from_array(time, name="time", label="time")

        return cls(data, time, name, label, unit)


class QuantityArray(Quantity):
    def __init__(
        self, data: da.Array, time: Axis, name: str, label: str, unit: str
    ) -> None:
        if data.ndim != 1:
            raise ValueError("only 1d data is supported")

        if data.dtype.fields:
            raise ValueError("only unstructured data types supported")

        self._name = name
        self._label = label
        self._unit = unit

        self._num = data.shape[0]
        self._time = time

        self._data = data

    def __getitem__(self, key: Any) -> Union["Quantity", "QuantityArray"]:
        raise NotImplementedError


class Particle(Quantity):
    def __init__(
        self, data: da.Array, time: Axis, name: str, label: str, unit: str
    ) -> None:
        if data.ndim != 0:
            raise ValueError("only 0d data is supported")

        if not data.dtype.fields:
            raise ValueError("only structured data types supported")

        self._name = name
        self._label = label
        self._unit = unit

        self._num = 1
        self._time = time

        self._data = data

    def __getitem__(self, key: Any) -> Union["Quantity", "ParticleArray"]:
        raise NotImplementedError


class ParticleArray(QuantityArray):
    def __init__(
        self, data: da.Array, time: Axis, name: str, label: str, unit: str
    ) -> None:
        if data.ndim != 1:
            raise ValueError("only 1d data is supported")

        if not data.dtype.fields:
            raise ValueError("only structured data types supported")

        self._name = name
        self._label = label
        self._unit = unit

        self._num = data.shape[0]
        self._time = time

        self._data = data

    def __getitem__(
        self, key: Any
    ) -> Union["Quantity", "QuantityArray", "ParticleArray"]:
        raise NotImplementedError


class ParticleDataset:
    pass
