# -*- coding: utf-8 -*-
from pathlib import Path
from textwrap import dedent
from typing import Any
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import runtime_checkable

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike

from nata.utils.types import BasicIndexing
from nata.utils.types import FileLocation

from .axis import Axis
from .axis import HasTimeAxis
from .core import HasBackends
from .core import HasName
from .core import HasNumpyInterface
from .core import HasParticleCount
from .core import HasPluginSystem
from .core import HasQuantities
from .core import HasUnit
from .exceptions import NoValidBackend
from .utils import unstructured_to_structured


@runtime_checkable
class ParticleBackend(Protocol):
    name: str
    location: Path

    def __init__(self, location: FileLocation) -> None:
        ...

    @staticmethod
    def is_valid_backend(location: FileLocation) -> bool:
        ...

    dataset_name: str
    dataset_label: str

    num_particles: int

    quantity_names: Sequence[str]
    quantity_labels: Sequence[str]
    quantity_units: Sequence[str]

    iteration: int
    time_step: float
    time_unit: str

    shape: Tuple[int, ...]
    dtype: np.dtype
    ndim: int


@runtime_checkable
class ParticleDataReader(ParticleBackend, Protocol):
    def get_data(
        self,
        indexing: Optional[BasicIndexing] = None,
        fields: Optional[Union[str, Sequence[str]]] = None,
    ) -> np.ndarray:
        ...


class Quantity(
    HasNumpyInterface,
    HasName,
    HasUnit,
    HasPluginSystem,
    HasTimeAxis,
    HasParticleCount,
):
    def __init__(
        self,
        data: da.Array,
        time: Axis,
        name: str,
        label: str,
        unit: str,
    ) -> None:
        if data.ndim != 0:
            raise ValueError("only 0d data is supported")

        if data.dtype.fields:
            raise ValueError("only unstructured data types supported")

        self._name = name
        self._label = label
        self._unit = unit

        self._count = 1
        self._time = time

        self._data = data

    def __getitem__(self, key: Any) -> Union["Quantity", "QuantityArray"]:
        raise NotImplementedError

    def __hash__(self) -> int:
        # general naming
        key = (self._name, self._label, self._unit, self._data.shape)

        # time specific
        key += (self._time.name, self._time.label, self._time.unit)

        # number of particles
        key += (self._count,)

        return hash(key)

    def __repr__(self) -> str:
        repr_ = (
            f"{type(self).__name__}<"
            f"name={self.name}, "
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
        | **count** | {self.count} |
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
        self,
        data: da.Array,
        time: Axis,
        name: str,
        label: str,
        unit: str,
    ) -> None:
        if not data.ndim >= 1:
            raise ValueError("only 1d and higher dimensional data is supported")

        if data.dtype.fields:
            raise ValueError("only unstructured data types supported")

        self._name = name
        self._label = label
        self._unit = unit

        self._count = data.shape[-1]
        self._time = time

        self._data = data

    def __getitem__(self, key: Any) -> Union["Quantity", "QuantityArray"]:
        raise NotImplementedError


class Particle(
    HasNumpyInterface,
    HasQuantities,
    HasPluginSystem,
    HasName,
    HasTimeAxis,
    HasParticleCount,
):
    def __init__(
        self,
        data: da.Array,
        quantities: Tuple[Tuple[str, str, str], ...],
        time: Axis,
        name: str,
        label: str,
    ) -> None:
        if data.ndim != 0:
            raise ValueError("only 0d data is supported")

        if not data.dtype.fields:
            raise ValueError("only structured data types supported")

        if (len(data.dtype.fields) != len(quantities)) or any(
            q[0] != f for q, f in zip(quantities, data.dtype.fields)
        ):
            raise ValueError("mismatch betweend fields and quantities")

        self._data = data
        self._quantities = quantities

        self._name = name
        self._label = label

        self._time = time
        self._count = 1

    def __getitem__(self, key: Any) -> Union["Quantity", "ParticleArray"]:
        raise NotImplementedError

    def __hash__(self) -> int:
        # general naming
        key = (self._name, self._label)

        # data properties
        key += (self._data.shape, self._data.dtype)

        # time specific
        key += (self._time.name, self._time.label, self._time.unit)

        # number of particles
        key += (self._count,)

        # quantities
        for q in self._quantities:
            key += q

        return hash(key)

    def __repr__(self) -> str:
        repr_ = (
            f"{type(self).__name__}<"
            f"quantities={self.quantity_names}, "
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
        | **count** | {self.count} |
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
        quantities: Optional[Tuple[Tuple[str, str, str], ...]] = None,
        time: Optional[Union[Axis, int, float]] = None,
    ) -> "Particle":
        if time is None:
            time = Axis.from_array(0.0, name="time", label="time")
        else:
            if not isinstance(time, Axis):
                time = Axis.from_array(time, name="time", label="time")

        if not isinstance(data, da.Array):
            data = da.asanyarray(data)

        if quantities is None:
            if data.dtype.fields:
                quantities = tuple((f, f, "") for f in data.dtype.fields)
            else:
                quantities = (("quant1", "quant1 label", ""),)
                new_dtype = np.dtype([("quant1", data.dtype)])
                data = unstructured_to_structured(data[..., np.newaxis], new_dtype)
        else:
            if not data.dtype.fields:
                new_dtype = np.dtype([(q[0], data.dtype) for q in quantities])
                data = unstructured_to_structured(data, new_dtype)

        return cls(data, quantities, time, name, label)


class ParticleArray(
    Particle,
    HasBackends,
    backend_protocol=ParticleBackend,
):
    def __init__(
        self,
        data: da.Array,
        quantities: Tuple[Tuple[str, str, str], ...],
        time: Axis,
        name: str,
        label: str,
    ) -> None:
        if not data.ndim >= 1:
            raise ValueError("0d data is not supported")

        if not data.dtype.fields:
            raise ValueError("only structured data types supported")

        if (len(data.dtype.fields) != len(quantities)) or any(
            q[0] != f for q, f in zip(quantities, data.dtype.fields)
        ):
            raise ValueError("mismatch betweend fields and quantities")

        self._data = data
        self._quantities = quantities

        self._name = name
        self._label = label

        self._time = time
        self._count = data.shape[-1]

    def __getitem__(
        self, key: Any
    ) -> Union["Quantity", "QuantityArray", "ParticleArray"]:
        raise NotImplementedError

    @staticmethod
    def _unpack_backend(
        backend: ParticleBackend, path: Path, time_axis: str
    ) -> Tuple[da.Array, Tuple[Tuple[str, str, str], ...], Axis, str, str]:
        prts = backend(path)
        data = da.from_array(prts)

        name = prts.dataset_name
        label = prts.dataset_label

        if time_axis == "time":
            time = Axis.from_array(
                prts.time_step,
                name="time",
                label="time",
                unit=prts.time_unit,
            )
        else:
            time = Axis.from_array(
                prts.iteration,
                name="iteration",
                label="iteration",
            )

        quantities = tuple(
            zip(prts.quantity_names, prts.quantity_labels, prts.quantity_units)
        )

        return data, quantities, time, name, label

    @classmethod
    def from_path(
        cls, path: Union[str, Path], time_axis: str = "time"
    ) -> "ParticleArray":
        if not isinstance(path, Path):
            path = Path(path)

        # check validity of passed arguments
        if not path.is_file():
            raise ValueError("only a single file is supported")

        if time_axis not in ("time", "iteration"):
            raise ValueError("only 'time' and 'iteration' are supported for time axis")

        # get valid backend
        backend = cls.get_valid_backend(path)
        if not backend:
            raise NoValidBackend(f"no valid backend for '{path}' found")

        backend_data = cls._unpack_backend(backend, path, time_axis)
        return cls(*backend_data)


class ParticleDataset(
    HasNumpyInterface,
    HasQuantities,
    HasPluginSystem,
    HasName,
    HasTimeAxis,
    HasParticleCount,
):
    def __init__(
        self,
        data: da.Array,
        quantities: Tuple[Tuple[str, str, str], ...],
        time: Axis,
        name: str,
        label: str,
    ) -> None:
        if data.ndim != 2:
            raise ValueError("only 2d data is supported")

        if not data.dtype.fields:
            raise ValueError("only structured data types supported")

        if (len(data.dtype.fields) != len(quantities)) or any(
            q[0] != f for q, f in zip(quantities, data.dtype.fields)
        ):
            raise ValueError("mismatch betweend fields and quantities")

        if time.ndim != 1:
            raise ValueError("only 1d time axis are supported")

        self._data = data
        self._quantities = quantities

        self._name = name
        self._label = label

        self._time = time
        self._count = data.shape[-1]

    def __hash__(self) -> int:
        # general naming
        key = (self.name, self.label)

        # data props
        key += (self.shape,)

        # time
        key += (self.time.name, self.time.label, self.time.unit)

        return hash(key)

    def __repr__(self) -> str:
        repr_ = (
            f"{type(self).__name__}"
            "<"
            f"name={self.name}, "
            f"dtype={self.dtype}, "
            f"quantities={self.quantity_names}, "
            f"time={repr(self.time)}"
            ">"
        )
        return repr_

    def _repr_markdown_(self) -> str:
        md = f"""
        | **{type(self).__name__}** | |
        | ---: | :--- |
        | **name**       | {self.name} |
        | **label**      | {self.label} |
        | **count**      | {self.count} |
        | **shape**      | {self.shape} |
        | **dtype**      | {self.dtype} |
        | **quantities** | {self.quantity_names} |
        | **time**       | {self.time} |

        """
        return dedent(md)

    def __getitem__(self, key: Any) -> Union["ParticleDataset", "ParticleArray"]:
        raise NotImplementedError

    @classmethod
    def from_array(
        cls,
        data: da.Array,
        *,
        name: str = "unnamed",
        label: str = "unlabeled",
        quantities: Optional[Tuple[Tuple[str, str, str], ...]] = None,
        time: Optional[Union[Axis, int, float]] = None,
    ) -> "ParticleDataset":
        if not isinstance(data, da.Array):
            data = da.asanyarray(data)

        if data.ndim not in (2, 3):
            raise ValueError("data array has to be at least 2d")

        if not data.dtype.fields:
            if data.ndim == 2:
                data = data[..., np.newaxis]

            dtype_list = [(f"quant{i+1}", data.dtype) for i in range(data.shape[-1])]
            new_dtype = np.dtype(dtype_list)

            data = unstructured_to_structured(data, new_dtype)

        if time is None:
            time = Axis.from_array(da.arange(len(data)), name="time", label="time")
        else:
            if not isinstance(time, Axis):
                time = Axis.from_array(time, name="time", label="time")

        if quantities is None:
            quantities = tuple((f, f"{f} label", "") for f in data.dtype.fields)

        return cls(data, quantities, time, name, label)
