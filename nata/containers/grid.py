# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union

import attr
import numpy as np
from attr.validators import deep_iterable
from attr.validators import instance_of
from attr.validators import optional

from nata.backends.grid import GridArray
from nata.backends.grid import GridBackend
from nata.utils.attrs import attrib_equality
from nata.utils.attrs import subdtype_of

from .axes import GridAxis
from .axes import IterationAxis
from .axes import TimeAxis
from .base import BaseDataset
from .base import convert_unstructured_data_to_array

BackendBased = TypeVar("BackendBased", str, Path, GridBackend)


@attr.s(init=False, eq=False)
class GridDataset(BaseDataset):
    """Container class storing grid datasets"""

    _backends: Set[GridBackend] = set((GridArray,))
    backend: Optional[str] = attr.ib(validator=optional(subdtype_of(np.str_)))
    appendable = True

    name: str = attr.ib(validator=subdtype_of(np.str_))
    label: str = attr.ib(validator=subdtype_of(np.str_))
    unit: str = attr.ib(validator=subdtype_of(np.str_))

    iteration: IterationAxis = attr.ib(validator=instance_of(IterationAxis))
    time: TimeAxis = attr.ib(validator=instance_of(TimeAxis))

    axes: List[GridAxis] = attr.ib(
        validator=deep_iterable(
            member_validator=instance_of(GridAxis),
            iterable_validator=instance_of(list),
        )
    )

    _data: Union[np.ndarray, List[Union[np.ndarray, GridBackend]]] = attr.ib(
        repr=False
    )
    dtype: np.dtype = attr.ib(validator=instance_of(np.dtype))
    data_shape: Tuple[int] = attr.ib(
        validator=deep_iterable(
            member_validator=subdtype_of(np.integer),
            iterable_validator=instance_of(tuple),
        )
    )
    data_ndim: int = attr.ib(validator=subdtype_of(np.integer))

    def info(self, full: bool = False):  # pragma: no cover
        return self.__repr__()

    def __init__(self, grid: Optional[BackendBased], **kwargs: Dict[str, Any]):
        if grid is None:
            self._init_from_kwargs(**kwargs)
        else:
            self._init_from_backend(grid)

        attr.validate(self)

    def _init_from_kwargs(self, **kwargs: Dict[str, Any]):
        for prop in (
            "backend",
            "name",
            "label",
            "unit",
            "iteration",
            "time",
            "axes",
            "_data",
            "data_shape",
            "data_ndim",
            "dtype",
        ):
            setattr(self, prop, kwargs[prop])

        for axis in kwargs["axes"]:
            setattr(self, axis.name, axis)

    def _init_from_backend(self, grid: BackendBased):
        if not isinstance(grid, GridBackend):
            grid = self._convert_to_backend(grid)

        self.backend = grid.name

        self.name = grid.dataset_name
        self.label = grid.dataset_label
        self.unit = grid.dataset_unit

        self.iteration = IterationAxis(grid.iteration)
        self.time = TimeAxis(grid.time_step, unit=grid.time_unit)

        self.axes = []

        for (name, label, unit, min_, max_, length) in zip(
            grid.axes_names,
            grid.axes_labels,
            grid.axes_units,
            grid.axes_min,
            grid.axes_max,
            grid.shape,
        ):
            axis = GridAxis(
                (min_, max_),
                axis_length=length,
                name=name,
                label=label,
                unit=unit,
            )
            setattr(self, name, axis)
            self.axes.append(axis)

        self.data_shape = grid.shape
        self.data_ndim = grid.dim
        self.dtype = grid.dtype

        self._data = [grid]
        self._data_indices = None

    def __eq__(self, other: Union["GridDataset", Any]):
        if not isinstance(other, self.__class__):
            return False

        if not attrib_equality(self, other, "name, label, unit"):
            return False

        if self.iteration != other.iteration and self.time != other.time:
            return False

        for axis in self.axes:
            if not hasattr(other, axis.name):
                return False

            if axis != getattr(other, axis.name):
                return False

        return True

    def __array__(self, dtype=None):
        if not isinstance(self._data, np.ndarray):
            self._data = convert_unstructured_data_to_array(
                self._data, self.dtype, self._data_indices
            )

        return self._data

    def __len__(self):
        return len(self.iteration)

    @property
    def shape(self):
        if len(self) == 1:
            return self.data_shape
        else:
            return (len(self),) + self.data_shape

    @property
    def ndim(self):
        return len(self.shape)

    @classmethod
    def from_array(
        cls,
        array,
        name=None,
        label=None,
        unit=None,
        iteration=None,
        time=None,
        time_unit=None,
        axes_names=None,
        axes_min=None,
        axes_max=None,
        axes_labels=None,
        axes_units=None,
        slice_array=True,
    ):
        try:
            time_iter = iter(time)
        except TypeError:
            time_iter = None

        try:
            iteration_iter = iter(iteration)
        except TypeError:
            iteration_iter = None

        if (time_iter is None) and (iteration_iter is None):
            if iteration is None:
                iteration = 0
            if time is None:
                time = 0.0

            arr_backend = GridArray(
                array=array,
                dataset_name=name,
                dataset_label=label,
                dataset_unit=unit,
                iteration=iteration,
                time_step=time,
                time_unit=time_unit,
                axes_names=axes_names,
                axes_min=axes_min,
                axes_max=axes_max,
                axes_labels=axes_labels,
                axes_units=axes_units,
            )

            return cls(arr_backend)

        if len(time) != len(iteration):
            raise ValueError(
                "mismatched in length between parameters time and iteration"
            )

        if slice_array:
            if array.shape[0] != len(time):
                raise ValueError("mismatch in iteration and array dimension")
            arr_index = 0
        else:
            arr_index = slice(None)

        t = next(time_iter)
        it = next(iteration_iter)

        ds = cls(
            GridArray(
                array=array[arr_index],
                dataset_name=name,
                dataset_label=label,
                dataset_unit=unit,
                iteration=it,
                time_step=t,
                time_unit=time_unit,
                axes_names=axes_names,
                axes_min=axes_min,
                axes_max=axes_max,
                axes_labels=axes_labels,
                axes_units=axes_units,
            )
        )
        for i, (t, it) in enumerate(zip(time_iter, iteration_iter), start=1):
            if slice_array:
                arr_index = i
            else:
                arr_index = slice(None)

            ds.append(
                cls(
                    GridArray(
                        array=array[arr_index],
                        dataset_name=name,
                        dataset_label=label,
                        dataset_unit=unit,
                        iteration=it,
                        time_step=t,
                        time_unit=time_unit,
                        axes_names=axes_names,
                        axes_min=axes_min,
                        axes_max=axes_max,
                        axes_labels=axes_labels,
                        axes_units=axes_units,
                        keep_creation_count=True,
                    )
                )
            )

        return ds

    def append(self, other: "GridDataset"):
        self._check_appendability(other)

        if isinstance(self._data, np.ndarray):
            if isinstance(other._data, np.ndarray):
                self._data = np.stack([self._data, other._data])
            else:
                if len(self) == 1:
                    self._data = [self._data] + [d for d in other._data]
                else:
                    self._data = [d for d in self._data]
                    self._data += [d for d in other._data]
        else:
            if isinstance(other._data, np.ndarray):
                if len(other) == 1:
                    self._data.append(other._data)
                else:
                    for d in other._data:
                        self._data.append(d)
            else:
                for d in other._data:
                    self._data.append(d)

        self.iteration.append(other.iteration)
        self.time.append(other.time)
        for axis, other_axis in zip(self.axes, other.axes):
            axis.append(other_axis)
