from pathlib import Path
from typing import Dict
from typing import Set
from typing import Tuple
from typing import List
from typing import ValuesView
from typing import Optional
from typing import Union
from copy import copy

import attr
from attr.validators import instance_of
from attr.validators import deep_iterable
from attr.validators import optional
import numpy as np

from nata.containers.base import BaseDataset
from nata.backends.grid import BaseGrid
from nata.backends.grid import GridArray

from nata.utils.exceptions import NataInvalidContainer

from .axes import IterationAxis
from .axes import TimeAxis
from .axes import GridAxis
from .axes import DataStock

_incomparable = {"eq": False, "order": False}


@attr.s(init=False)
class GridDataset(BaseDataset):
    _backends: Set[BaseGrid] = set()
    backend: Optional[str] = attr.ib(validator=optional(instance_of(str)))
    appendable = True

    name: str = attr.ib(validator=instance_of(str))
    label: str = attr.ib(validator=instance_of(str))
    unit: str = attr.ib(validator=instance_of(str))

    iteration: IterationAxis = attr.ib(validator=instance_of(IterationAxis))
    time: TimeAxis = attr.ib(validator=instance_of(TimeAxis))

    axes: List[GridAxis] = attr.ib(
        validator=deep_iterable(
            member_validator=instance_of(GridAxis),
            iterable_validator=instance_of(list),
        ),
        repr=lambda values: "[" + ", ".join(v.name for v in values) + "]",
    )

    grid_shape: Tuple[int] = attr.ib(
        validator=deep_iterable(
            member_validator=instance_of(int),
            iterable_validator=instance_of(tuple),
        )
    )
    grid_dim: int = attr.ib(validator=instance_of(int))
    grid_dtype: np.dtype = attr.ib(validator=instance_of((type, np.dtype)))

    _data: DataStock = attr.ib(repr=False, **_incomparable)

    def info(self, full: bool = False):  # pragma: no cover
        return self.__repr__()

    def __init__(self, gridobj: Union[str, Path, BaseGrid]):
        if not isinstance(gridobj, BaseGrid):
            for backend in self._backends:
                if backend.is_valid_backend(gridobj):
                    gridobj = backend(gridobj)
                    break
            else:
                raise NataInvalidContainer

        self.backend = gridobj.name
        self.location = gridobj.location
        self.name = gridobj.dataset_name
        self.label = gridobj.dataset_label
        self.unit = gridobj.dataset_unit

        self.iteration = IterationAxis(
            parent=self, key=gridobj.iteration, value=gridobj.iteration
        )
        self.time = TimeAxis(
            parent=self,
            key=gridobj.iteration,
            value=gridobj.time_step,
            unit=gridobj.time_unit,
        )
        self._data = DataStock(
            key=gridobj.iteration,
            value=gridobj,
            shape=gridobj.shape,
            dtype=gridobj.dtype,
        )
        self.grid_shape = self._data.shape
        self.grid_dim = self._data.dim
        self.grid_dtype = self._data.dtype

        self.axes = []

        for (name, label, unit, min_, max_, length) in zip(
            gridobj.axes_names,
            gridobj.axes_labels,
            gridobj.axes_units,
            gridobj.axes_min,
            gridobj.axes_max,
            gridobj.shape,
        ):
            axis = GridAxis(
                parent=self,
                key=gridobj.iteration,
                value=[min_, max_],
                name=name,
                label=label,
                unit=unit,
                length=length,
            )
            setattr(self, name, axis)
            self.axes.append(axis)

        self._step = None

        attr.validate(self)

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
        if self != other:
            raise ValueError(f"can not append '{other}' to '{self}'")

        self.iteration.update(other.iteration)
        self.time.update(other.time)
        for axis, other_axis in zip(self.axes, other.axes):
            axis.update(other_axis)
        self._data.update(other._data)

    def __len__(self):
        return len(self.iteration)

    @property
    def data(self) -> np.ndarray:
        if self._step is None:
            return self._data[:]
        else:
            return self._data[self._step]

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        if self._step is None:
            self._data[:] = new_data
        else:
            self._data[self._step] = new_data

    def copy(self):
        # TODO: check that axis parents reference new copy
        return copy(self)

    def iter(self, with_iteration=True):
        for step in self.iteration.keys():
            self._step = step
            if with_iteration:
                yield self._step, self
            else:
                yield self
        self._step = None

    def __iter__(self):
        for step in self.iteration.keys():
            self._step = step
            yield self
        self._step = None
