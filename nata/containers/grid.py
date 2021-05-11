# -*- coding: utf-8 -*-
from collections import defaultdict
from pathlib import Path
from textwrap import dedent
from typing import Any
from typing import Dict
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
from .axis import HasAxes
from .axis import HasTimeAxis
from .core import HasAnnotations
from .core import HasBackends
from .core import HasNumpyInterface
from .core import HasPluginSystem
from .exceptions import NoValidBackend
from .utils import is_unique

__all__ = ["GridArray", "GridDataset", "stack"]


def stack(grid_arrs: Sequence["GridArray"]) -> "GridDataset":
    if not len(grid_arrs):
        raise ValueError("can not iterate over 0-length sequence of GridArrays")

    if not is_unique(hash(grid) for grid in grid_arrs):
        raise ValueError("provided GridArrays are not equivalent to each other")

    base = grid_arrs[0]

    name = base.name
    label = base.label
    unit = base.unit

    data = da.stack([grid.to_dask() for grid in grid_arrs])
    time = Axis.from_array(
        da.stack([grid.time.to_dask() for grid in grid_arrs]),
        name=base.time.name,
        label=base.time.label,
        unit=base.time.unit,
    )

    axes = []
    for i, ax in enumerate(base.axes):
        axes_data = da.stack([grid.axes[i].to_dask() for grid in grid_arrs])
        axes.append(
            Axis.from_array(axes_data, name=ax.name, label=ax.label, unit=ax.unit)
        )

    return GridDataset(data, (time,) + tuple(axes), name=name, label=label, unit=unit)


@runtime_checkable
class GridBackend(Protocol):
    name: str
    location: Path

    def __init__(self, location: FileLocation) -> None:
        ...

    @staticmethod
    def is_valid_backend(location: FileLocation) -> bool:
        ...

    dataset_name: str
    dataset_label: str
    dataset_unit: str

    axes_names: Sequence[str]
    axes_labels: Sequence[str]
    axes_units: Sequence[str]
    axes_min: np.ndarray
    axes_max: np.ndarray

    iteration: int
    time_step: float
    time_unit: str

    shape: Tuple[int, ...]
    dtype: np.dtype
    ndim: int


@runtime_checkable
class GridDataReader(GridBackend, Protocol):
    def get_data(self, indexing: Optional[BasicIndexing] = None) -> np.ndarray:
        ...


class GridArray(
    HasBackends,
    HasAnnotations,
    HasNumpyInterface,
    HasPluginSystem,
    HasAxes,
    HasTimeAxis,
    backend_protocol=GridBackend,
):
    def __init__(
        self,
        data: da.Array,
        axes: Tuple[Axis, ...],
        time: Axis,
        name: str,
        label: str,
        unit: str,
    ) -> None:
        # name property has to be a valid identifier
        if not name.isidentifier():
            raise ValueError("'name' has to be a valid identifier")

        # time axis has to be unsized
        if time.ndim != 0:
            raise ValueError("time axis has to be 0 dimensional")

        # ensure that every dimension has a corresponding axis
        if len(axes) != data.ndim:
            raise ValueError("number of axes mismatches with dimensionality of data")

        # ensure consistency between .data and .axes
        for ax, s in zip(axes, data.shape):
            if ax.ndim != 1:
                raise ValueError("only 1D axis for GridArray are supported")

            if ax.shape[0] != s:
                raise ValueError("inconsistency between data and axis shape")

        self._data = data
        self._axes = axes
        self._time = time
        self._name = name
        self._label = label
        self._unit = unit

    def __getitem__(self, key: Any) -> "GridArray":
        index = ndx.ndindex(key).expand(self.shape).raw

        new_axis_excluded = tuple(ind for ind in index if ind is not None)
        indices_of_new_axis = tuple(i for i, ind in enumerate(index) if ind is None)
        reductions = tuple(isinstance(idx, int) for idx in new_axis_excluded)

        axes = []
        for ax, ind, red in zip(self.axes, new_axis_excluded, reductions):
            if not red:
                axes.append(ax[ind])

        for pos in indices_of_new_axis:
            axes.insert(pos, Axis.from_array([0]))

        data = self._data[index]
        axes = tuple(axes)
        time = self.time
        name = self.name
        label = self.label
        unit = self.unit

        return GridArray(data, axes, time, name, label, unit)

    def __hash__(self) -> int:
        # general naming
        key = (self.name, self.label, self.unit, self.shape)

        # time specific
        key += (self.time.name, self.time.label, self.time.unit)

        # axes specific
        for ax in self.axes:
            key += (ax.name, ax.label, ax.unit)

        return hash(key)

    def __repr__(self) -> str:
        repr_ = (
            f"{type(self).__name__}<"
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"time={self.time.to_numpy()}, "
            f"axes=({', '.join(f'Axis({ax.name})' for ax in self.axes)})"
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
        | **shape** | {self.shape} |
        | **dtype** | {self.dtype} |
        | **time**  | {self.time.to_numpy()} |
        | **axes**  | {', '.join(f"Axis({ax.name})" for ax in self.axes)} |

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
        axes: Optional[Sequence[ArrayLike]] = None,
        time: Optional[Union[Axis, int, float]] = None,
    ) -> "GridArray":
        if not isinstance(data, da.Array):
            data = da.asanyarray(data)

        if axes is None:
            axes = ()
            for i, l in enumerate(data.shape):
                axes += (Axis.from_array(da.arange(l), name=f"axis{i}"),)
        else:
            # ensure that every element in axes is an axis
            if any(not isinstance(ax, Axis) for ax in axes):
                tmp = []
                for i, ax in enumerate(axes):
                    if not isinstance(ax, Axis):
                        ax = Axis.from_array(da.asanyarray(ax), name=f"axis{i}")
                    tmp.append(ax)

                axes = tuple(tmp)

        if time is None:
            time = Axis.from_array(0.0, name="time", label="time")
        else:
            if not isinstance(time, Axis):
                time = Axis.from_array(time, name="time", label="time")

        return cls(data, axes, time, name, label, unit)

    @staticmethod
    def _unpack_backend(
        backend: GridBackend, path: Path, time_axis: str
    ) -> Tuple[da.Array, Tuple[Axis, ...], Axis, str, str, str]:
        grid = backend(path)
        data = da.from_array(grid)

        name = grid.dataset_name
        label = grid.dataset_label
        unit = grid.dataset_unit

        if time_axis == "time":
            time = Axis.from_array(
                grid.time_step, name="time", label="time", unit=grid.time_unit
            )
        else:
            time = Axis.from_array(grid.iteration, name="iteration", label="iteration")

        axes = ()
        for min_, max_, ax_pts, ax_name, ax_label, ax_unit in zip(
            grid.axes_min,
            grid.axes_max,
            grid.shape,
            grid.axes_names,
            grid.axes_labels,
            grid.axes_units,
        ):
            axes += (
                Axis.from_limits(
                    min_, max_, ax_pts, name=ax_name, label=ax_label, unit=ax_unit
                ),
            )

        return data, axes, time, name, label, unit

    @classmethod
    def from_path(cls, path: Union[str, Path], time_axis: str = "time") -> "GridArray":
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

        return cls(*cls._unpack_backend(backend, path, time_axis))


class GridDataset(
    HasAnnotations,
    HasNumpyInterface,
    HasPluginSystem,
    HasAxes,
    HasTimeAxis,
):
    def __init__(
        self,
        data: da.Array,
        axes: Tuple[Axis, ...],
        name: str,
        label: str,
        unit: str,
    ) -> None:
        # name property has to be a valid identifier
        if not name.isidentifier():
            raise ValueError("'name' has to be a valid identifier")

        # ensure that every dimension has a corresponding axis
        if len(axes) != data.ndim:
            raise ValueError("number of axes mismatches with dimensionality of data")

        # ensure consistency between .data and .axes, e.g.
        # data.shape = (5, 6, 7)
        #    > axes[0].shape = (5,)        =|> corresponds to 'time' axis
        #    > axes[1].shape = (5, 6)      =|> corresponds to 'axis0' axis
        #    > axes[2].shape = (5, 7)      =|> corresponds to 'axis1' axis
        for i, (ax, s) in enumerate(zip(axes, data.shape)):
            if i == 0:
                if ax.ndim != 1:
                    raise ValueError("time axis has to be 1D")
                if ax.shape[0] != s:
                    raise ValueError("inconsistency between data and axis shape")
            else:
                if ax.ndim != 2:
                    raise ValueError("only 2D axis for GridDataset are supported")
                if ax.shape[1] != s:
                    raise ValueError("inconsistency between data and axis shape")

        self._data = data
        self._axes = axes
        self._time = axes[0]
        self._name = name
        self._label = label
        self._unit = unit

    def __hash__(self) -> int:
        # general naming
        key = (self.name, self.label, self.unit, self.shape)

        # time specific
        key += (self.time.name, self.time.label, self.time.unit)

        # axes specific
        for ax in self.axes:
            key += (ax.name, ax.label, ax.unit)

        return hash(key)

    def __repr__(self) -> str:
        repr_ = (
            f"{type(self).__name__}<"
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"time={self.time.to_numpy()}, "
            f"axes=({', '.join(f'Axis({ax.name})' for ax in self.axes)})"
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
        | **shape** | {self.shape} |
        | **dtype** | {self.dtype} |
        | **time**  | {self.time.to_numpy()} |
        | **axes**  | {', '.join(f"Axis({ax.name})" for ax in self.axes)} |

        """
        return dedent(md)

    def __getitem__(self, key: Any) -> Union["GridDataset", GridArray]:
        # unpack axes -> remember the first axis is responsible for time slicing
        time_axis, *rest_axes = self.axes

        # unpack indexing
        index = ndx.ndindex(key).expand(self.shape).raw
        time_slicing, *axes_slicing = index

        # nata does not support adding a new axis
        if time_slicing is np.newaxis:
            msg = (
                "creating a new axis as time axis is not supported\n"
                "use `.to_dask` and `.to_numpy` and convert to a GridDataset afterwards"
            )
            raise IndexError(msg)

        new_axis_excluded = tuple(ind for ind in axes_slicing if ind is not None)
        indices_of_new_axis = tuple(
            i for i, ind in enumerate(axes_slicing) if ind is None
        )
        reductions = tuple(isinstance(idx, int) for idx in new_axis_excluded)

        axes = [time_axis[time_slicing]]
        for ax, ind, red in zip(rest_axes, new_axis_excluded, reductions):
            if not red:
                axes.append(ax[time_slicing, ind])

        for pos in indices_of_new_axis:
            axes.insert(pos + 1, Axis.from_array(da.zeros((len(time_axis), 1))))

        data = self._data[index]
        name = self.name
        label = self.label
        unit = self.unit

        if axes[0].shape:
            return GridDataset(data, tuple(axes), name, label, unit)
        else:
            time, *axes = axes
            return GridArray(data, tuple(axes), time, name, label, unit)

    @classmethod
    def from_array(
        cls,
        data: ArrayLike,
        *,
        name: str = "unnamed",
        label: str = "unlabeled",
        unit: str = "",
        axes: Optional[Sequence[ArrayLike]] = None,
    ) -> "GridDataset":
        if not isinstance(data, da.Array):
            data = da.asanyarray(data)

        if axes is None:
            axes = ()
            time_steps = None
            for i, l in enumerate(data.shape):
                if i == 0:
                    time_steps = l
                    time = Axis.from_array(
                        da.arange(time_steps), name="time", label="time"
                    )
                    axes += (time,)
                else:
                    axis_shape = (time_steps, 1)
                    axis = Axis.from_array(
                        da.tile(da.arange(l), axis_shape), name=f"axis{i-1}"
                    )
                    axes += (axis,)

        else:
            # ensure that every element in axes is an axis
            if any(not isinstance(ax, Axis) for ax in axes):
                tmp = []

                for i, ax in enumerate(axes):
                    name = "time" if i == 0 else f"axis{i-1}"
                    label = "time" if i == 0 else "unlabeled"

                    if not isinstance(ax, Axis):
                        ax = Axis.from_array(da.asanyarray(ax), name=name, label=label)

                    tmp.append(ax)

                axes = tuple(tmp)

        return cls(data, axes, name, label, unit)

    @classmethod
    def from_path(
        cls, path: Union[str, Path], time_axis: str = "time"
    ) -> "GridDataset":
        files = FileList(path, recursive=False)

        # go over all the files and
        grid_arrs: Dict[int, List[GridArray]] = defaultdict(list)
        for f in files.paths:
            try:
                grid_arr = GridArray.from_path(f, time_axis=time_axis)
            except NoValidBackend:
                continue

            grid_arrs[hash(grid_arr)].append(grid_arr)

        # convert dict of
        grids = list(grid_arrs.values())

        if not len(grids):
            raise ValueError(f"no valid grid found for '{path}'")

        # warn if multiple grids were found
        if len(grids) > 1:
            warn(f"found multiple grids and picking grid '{grids[0].name}'")

        return stack(grids[0])
