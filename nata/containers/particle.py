# -*- coding: utf-8 -*-
from collections import defaultdict
from pathlib import Path
from textwrap import dedent
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import runtime_checkable
from warnings import warn

import dask.array as da
import ndindex as ndx
import numpy as np
from numpy.typing import ArrayLike

from nata.utils.io import FileList
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
from .utils import is_unique
from .utils import unstructured_to_structured


def expand_arr_1d(arr: da.Array, required_shape: Tuple[int]) -> da.Array:
    missing = (required_shape[0] - arr.shape[0],)
    values = da.block([arr, da.zeros(missing, dtype=arr.dtype)])
    mask = da.block([da.zeros(arr.shape, dtype=bool), da.ones(missing, dtype=bool)])
    return da.ma.masked_array(values, mask=mask)


def expand_arr(arr: da.Array, required_shape: Tuple[int]) -> da.Array:
    if arr.ndim == 1:
        return expand_arr_1d(arr, required_shape)
    else:
        raise NotImplementedError(f"not implemented for 'ndim = {arr.ndim}'")


def expand_and_stack(data: Iterable) -> da.Array:
    arrays = [d if isinstance(d, da.Array) else da.asanyarray(d) for d in data]
    max_shape = max(arr.shape for arr in arrays)
    arrays = [expand_arr(arr, max_shape) for arr in arrays]
    return da.stack(arrays)


def stack(part_arrs: Sequence["ParticleArray"]) -> "ParticleDataset":
    if not len(part_arrs):
        raise ValueError("can not iterate over 0-length sequence of GridArrays")

    if not is_unique(hash(prt) for prt in part_arrs):
        raise ValueError("provided GridArrays are not equivalent to each other")

    base = part_arrs[0]

    data = expand_and_stack([grid.to_dask() for grid in part_arrs])
    time = Axis.from_array(
        da.stack([grid.time.to_dask() for grid in part_arrs]),
        name=base.time.name,
        label=base.time.label,
        unit=base.time.unit,
    )

    return ParticleDataset(data, base.quantities, time, base.name, base.label)


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
        key = (self.name, self.label)

        # data properties
        key += (self.dtype,)

        # time specific
        key += (self.time.name, self.time.label, self.time.unit)

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

        HasNumpyInterface.__init__(self, data)
        HasQuantities.__init__(self, quantities)
        HasName.__init__(self, name, label)
        HasTimeAxis.__init__(self, time)
        HasParticleCount.__init__(self, data.shape[-1])

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

    @staticmethod
    def _expand_key(
        key: Any,
        shape: Tuple[int, int],
    ) -> Tuple[Union[int, slice], Union[int, slice], Union[str, List[str]]]:
        # convert to tuple
        key = np.index_exp[key]

        if len(key) > 3:
            raise IndexError("too many indices provided")

        if len(key) == 3:
            expanded_key = ndx.ndindex(key[:2]).expand(shape).raw + (key[2],)
        else:
            expanded_key = ndx.ndindex(key).expand(shape).raw + ([],)

        return expanded_key

    @staticmethod
    def _determine_reduction(
        index: Tuple[Union[int, slice], Union[int, slice], Union[str, Sequence[str]]],
    ) -> Tuple[bool, bool, bool]:
        return (
            isinstance(index[0], int),
            isinstance(index[1], int),
            isinstance(index[2], str),
        )

    def _decay_to_Quantity(self, index: Tuple[int, int, str]) -> "Quantity":
        quantity_index = self.quantity_names.index(index[2])
        name, label, unit = self.quantities[quantity_index]

        return Quantity.from_array(
            self._data[index[0], index[1]][index[2]],
            name=name,
            label=label,
            unit=unit,
            time=self.time[index[0]],
        )

    def _decay_to_QuantityArray(self, index: Tuple[int, slice, str]) -> "QuantityArray":
        quantity_index = self.quantity_names.index(index[2])
        name, label, unit = self.quantities[quantity_index]

        return QuantityArray.from_array(
            self._data[index[0], index[1]][index[2]],
            name=name,
            label=label,
            unit=unit,
            time=self.time[index[0]],
        )

    def _decay_to_Particle(self, index: Tuple[int, int, Sequence[str]]) -> "Particle":
        # if sequence not empty -> iterate over picking up quantities
        if index[2]:
            new_quantities = ()
            for quant in index[2]:
                quant_index = self.quantity_names.index(quant)
                new_quantities += (self.quantities[quant_index],)
            data = self._data[index[0], index[1]][index[2]]
        else:
            new_quantities = self.quantities
            data = self._data[index[0], index[1]]

        return Particle.from_array(
            data,
            name=self.name,
            label=self.label,
            quantities=new_quantities,
            time=self.time[index[0]],
        )

    def _decay_to_ParticleArray(
        self,
        index: Tuple[int, slice, Sequence[str]],
    ) -> "ParticleArray":
        # if sequence not empty -> iterate over picking up quantities
        if index[2]:
            new_quantities = ()
            for quant in index[2]:
                quant_index = self.quantity_names.index(quant)
                new_quantities += (self.quantities[quant_index],)
            data = self._data[index[0], index[1]][index[2]]
        else:
            new_quantities = self.quantities
            data = self._data[index[0], index[1]]

        return ParticleArray.from_array(
            data,
            name=self.name,
            label=self.label,
            quantities=new_quantities,
            time=self.time[index[0]],
        )

    def _decay_to_ParticleDataset(
        self,
        index: Tuple[slice, Union[int, slice], Sequence[str]],
    ) -> "ParticleDataset":
        # if sequence not empty -> iterate over picking up quantities
        if index[2]:
            new_quantities = ()
            for quant in index[2]:
                quant_index = self.quantity_names.index(quant)
                new_quantities += (self.quantities[quant_index],)
            data = self._data[index[0], index[1]][index[2]]
        else:
            new_quantities = self.quantities
            data = self._data[index[0], index[1]]

        return ParticleDataset.from_array(
            data,
            name=self.name,
            label=self.label,
            quantities=new_quantities,
            time=self.time[index[0]],
        )

    def __getitem__(
        self, key: Any
    ) -> Union[
        "ParticleDataset",
        "ParticleArray",
        "Particle",
        "Quantity",
        "QuantityArray",
    ]:
        key = self._expand_key(key, self.shape)
        time_reduction, prt_reduction, quant_reduction = self._determine_reduction(key)

        if quant_reduction and time_reduction and prt_reduction:
            return self._decay_to_Quantity(key)
        elif quant_reduction and time_reduction:
            return self._decay_to_QuantityArray(key)
        elif time_reduction and prt_reduction:
            return self._decay_to_Particle(key)
        elif time_reduction:
            return self._decay_to_ParticleArray(key)
        else:
            return self._decay_to_ParticleDataset(key)

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
            data = expand_and_stack(data)

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

    @classmethod
    def from_path(
        cls, path: Union[str, Path], time_axis: str = "time"
    ) -> "ParticleDataset":
        files = FileList(path, recursive=False)

        prt_arrs: Dict[int, List[Particle]] = defaultdict(list)

        for f in files.paths:
            try:
                prt_arr = ParticleArray.from_path(f, time_axis=time_axis)
            except NoValidBackend:
                continue

            prt_arrs[hash(prt_arr)].append(prt_arr)

        parts = list(prt_arrs.values())

        if not len(parts):
            raise ValueError(f"no valid particles found for '{path}'")

        # warn if multiple grids were found
        if len(parts) > 1:
            warn(f"found multiple particles datasets and picking '{parts[0].name}'")

        return stack(parts[0])
