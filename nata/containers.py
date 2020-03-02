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
from nata.backends.particles import ParticleArray
from nata.backends.particles import ParticleBackend
from nata.utils.attrs import attrib_equality
from nata.utils.attrs import location_exists
from nata.utils.attrs import subdtype_of

from .axes import GridAxis
from .axes import IterationAxis
from .axes import ParticleQuantity
from .axes import TimeAxis
from .axes import UnnamedAxis
from .utils.exceptions import NataInvalidContainer


def register_backend(container):
    if not issubclass(container, BaseDataset):
        raise ValueError("Invalid container passed for backend registration!")

    def add_to_backend(backend):
        container.add_backend(backend)
        return backend

    return add_to_backend


class BaseDataset:
    _backends: Set[Any] = set()
    appendable = False

    @classmethod
    def register_plugin(cls, plugin_name, plugin):
        setattr(cls, plugin_name, plugin)

    @classmethod
    def add_backend(cls, backend):
        cls._backends.add(backend)

    def _convert_to_backend(self, obj):
        for backend in self._backends:
            if backend.is_valid_backend(obj):
                return backend(obj)

        raise NataInvalidContainer(
            f"Unable to find proper backend for {type(obj)}"
        )

    def _check_dataset_equality(self, other):
        raise NotImplementedError

    def _check_appendability(self, other: "BaseDataset"):
        if not self.appendable:
            raise TypeError(f"'{self.__class__}' is not appendable")

        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Can not append '{type(other)}' to '{self.__class__}'"
            )

        if not self._check_dataset_equality(other):
            raise ValueError(f"{other} can not be appended")


GridBackendBased = TypeVar("GridBackendBased", str, Path, GridBackend)


def _asanyarray_converter(data):
    data = np.asanyarray(data)

    if not data.ndim:
        data = data.reshape((1,))

    return data


def _deduct_prop_from_data(data, deduced_prop, default=None):
    prop_store = set()
    err_msg = f"Mixed data provided for property '{deduced_prop}'"

    for d in data:
        prop_store |= set((getattr(d, deduced_prop, None),))

    if len(prop_store) not in (1, 2):
        raise ValueError(err_msg)

    if len(prop_store) == 2:
        try:
            prop_store.remove(None)
        except KeyError:
            raise ValueError(err_msg)

    prop = prop_store.pop()
    return prop if prop is not None else default


def _generate_axes_list(data):
    axes_names = tuple()
    axes_labels = tuple()
    axes_units = tuple()
    axes_lengths = tuple()

    axes_min = []
    axes_max = []

    for d in data:
        if not hasattr(d, "axes_names"):
            continue

        names = tuple(d.axes_names)
        labels = tuple(d.axes_labels)
        units = tuple(d.axes_units)
        lengths = d.shape

        axes_min.append(getattr(d, "axes_min", None))
        axes_max.append(getattr(d, "axes_max", None))

        if any(tuple() == v for v in (axes_names, axes_labels, units, lengths)):
            axes_names = names
            axes_labels = labels
            axes_units = units
            axes_lengths = lengths
        else:
            if (names, labels, units, lengths) != (
                axes_names,
                axes_labels,
                axes_units,
                axes_lengths,
            ):
                raise ValueError("Mismatch between axes props in data")

    axes = []

    for name, label, unit, min_, max_, l in zip(
        axes_names, axes_labels, axes_units, axes_min, axes_max, axes_lengths
    ):
        axes.append(
            GridAxis(
                min_=min_,
                max_=max_,
                axis_length=l,
                name=name,
                label=label,
                unit=unit,
            )
        )

    return axes


@attr.s(eq=False)
class GridDataset(BaseDataset):
    """Container class storing grid datasets"""

    _backends: Set[GridBackend] = set((GridArray,))
    appendable = True

    _data: np.ndarray = attr.ib(converter=_asanyarray_converter, repr=False)
    _shape: Tuple[int] = attr.ib(
        default=None,
        validator=optional(
            deep_iterable(
                member_validator=instance_of(int),
                iterable_validator=instance_of(tuple),
            )
        ),
    )
    dtype: np.dtype = attr.ib(
        default=None, validator=optional(instance_of(np.dtype))
    )
    backend: Optional[str] = attr.ib(
        default=None, validator=optional(subdtype_of(np.str_))
    )

    name: str = attr.ib(default=None, validator=optional(subdtype_of(np.str_)))
    label: str = attr.ib(default=None, validator=optional(subdtype_of(np.str_)))
    unit: str = attr.ib(default=None, validator=optional(subdtype_of(np.str_)))

    iteration: IterationAxis = attr.ib(
        default=None, validator=optional(instance_of(IterationAxis))
    )
    time: TimeAxis = attr.ib(
        default=None, validator=optional(instance_of(TimeAxis))
    )

    axes: List[Optional[GridAxis]] = attr.ib(
        default=None,
        validator=optional(
            deep_iterable(
                member_validator=optional(instance_of(GridAxis)),
                iterable_validator=instance_of(list),
            )
        ),
    )

    # TODO: add dimensionality of the dataset

    def __attrs_post_init__(self):
        for i, d in enumerate(self._data):
            if not isinstance(d, (np.ndarray, GridBackend)):
                self._data[i] = self._convert_to_backend(d)

        self.dtype = (
            self.dtype
            if self.dtype is not None
            else _deduct_prop_from_data(self._data, "dtype")
        )
        self._shape = (
            self._shape
            if self._shape is not None
            else _deduct_prop_from_data(self._data, "shape")
        )

        self.name = (
            self.name
            if self.name is not None
            else _deduct_prop_from_data(self._data, "dataset_name")
        )
        self.label = (
            self.label
            if self.label is not None
            else _deduct_prop_from_data(self._data, "dataset_label")
        )
        self.unit = (
            self.unit
            if self.unit is not None
            else _deduct_prop_from_data(self._data, "dataset_unit")
        )

        self.iteration = (
            self.iteration
            if self.iteration is not None
            else IterationAxis(
                data=_deduct_prop_from_data(self._data, "iteration")
            )
        )
        self.time = (
            self.time
            if self.time is not None
            else TimeAxis(
                data=_deduct_prop_from_data(self._data, "time_step"),
                unit=_deduct_prop_from_data(self._data, "time_unit", ""),
            )
        )

        self.axes = (
            self.axes
            if self.axes is not None
            else _generate_axes_list(self._data)
        )

        for axis in self.axes:
            setattr(self, axis.name, axis)

        # cross validate - just for the ake for safety - we can remove later on
        attr.validate(self)

    def __iter__(self):
        if len(self) == 1:
            yield self
        else:
            for d, it, t in zip(self._data, self.iteration, self.time):
                yield self.__class__(
                    data=[d],
                    dtype=self.dtype,
                    backend=self.backend,
                    name=self.name,
                    label=self.label,
                    unit=self.unit,
                    iteration=it,
                    time=t,
                    axes=self.axes,
                )

    def __array__(self, dtype=None):
        if self._data.dtype == object:
            data = np.empty((len(self),) + self._shape, dtype=self.dtype)
            for i, d in enumerate(self._data):
                if isinstance(d, np.ndarray):
                    data[i] = d
                else:
                    data[i] = d.get_data(indexing=None)

            self._data = data

        return np.squeeze(self._data)

    def __len__(self):
        return len(self.iteration)

    def info(self, full: bool = False):  # pragma: no cover
        return self.__repr__()

    @property
    def shape(self):
        if self._data.dtype == object:
            if len(self) == 1:
                return self._shape
            else:
                return (len(self),) + self._shape
        else:
            return self._data.shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def grid_dim(self):
        return len(self.axes)

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

    def _check_dataset_equality(self, other: Union["GridDataset", Any]):
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

    def append(self, other: "GridDataset"):
        self._check_appendability(other)

        self._data = np.array(
            [d for d in self._data] + [d for d in other._data]
        )

        self.iteration.append(other.iteration)
        self.time.append(other.time)
        for axis, other_axis in zip(self.axes, other.axes):
            axis.append(other_axis)


ParticleBackendBased = TypeVar(
    "ParticleBackendBased", str, Path, ParticleBackend
)


@attr.s(init=False, eq=False)
class ParticleDataset(BaseDataset):
    """Container class storing particle datasets"""

    _backends: Set[ParticleBackend] = set((ParticleArray,))
    backend: Optional[str] = attr.ib(validator=optional(subdtype_of(np.str_)))
    appendable = True

    name: str = attr.ib(validator=subdtype_of(np.str_))
    dtype: np.dtype = attr.ib(validator=instance_of(np.dtype))

    iteration: IterationAxis = attr.ib(validator=instance_of(IterationAxis))
    time: TimeAxis = attr.ib(validator=instance_of(TimeAxis))
    num_particles: UnnamedAxis = attr.ib(validator=instance_of(UnnamedAxis))

    quantities: np.ndarray = attr.ib(
        validator=deep_iterable(
            member_validator=subdtype_of(np.str_),
            iterable_validator=instance_of(np.ndarray),
        ),
        eq=False,
    )
    quantity_labels: np.ndarray = attr.ib(
        validator=deep_iterable(
            member_validator=subdtype_of(np.str_),
            iterable_validator=instance_of(np.ndarray),
        ),
        eq=False,
    )
    quantity_units: np.ndarray = attr.ib(
        validator=deep_iterable(
            member_validator=subdtype_of(np.str_),
            iterable_validator=instance_of(np.ndarray),
        ),
        eq=False,
    )

    def info(self, full: bool = False):  # pragma: no cover
        return self.__repr__()

    def __init__(
        self,
        particles: Optional[ParticleBackendBased],
        **kwargs: Dict[str, Any],
    ):
        if particles is None:
            self._init_from_kwargs(**kwargs)
        else:
            self._init_from_backend(particles)

        attr.validate(self)

    def _init_from_kwargs(self, **kwargs: Dict[str, Any]):
        raise NotImplementedError

    def _init_from_backend(self, particles: ParticleBackendBased):
        if not isinstance(particles, ParticleBackend):
            particles = self._convert_to_backend(particles)

        self.backend = particles.name

        self.name = particles.dataset_name
        self.dtype = particles.dtype

        self.iteration = IterationAxis(data=particles.iteration)
        self.time = TimeAxis(data=particles.time_step, unit=particles.time_unit)
        self.num_particles = UnnamedAxis(data=particles.num_particles)

        self.quantities = np.asarray(particles.quantities)
        self.quantity_labels = np.asarray(particles.quantity_labels)
        self.quantity_units = np.asarray(particles.quantity_units)

        for quantity, label, unit in zip(
            self.quantities, self.quantity_labels, self.quantity_units
        ):
            q = ParticleQuantity(
                data=[particles],
                len=[particles.num_particles],
                dtype=self.dtype[quantity],
                name=quantity,
                label=label,
                unit=unit,
            )
            setattr(self, quantity, q)

    def _check_dataset_equality(self, other: Union["ParticleDataset", Any]):
        if not isinstance(other, self.__class__):
            return False

        if not attrib_equality(self, other):
            return False

        return True

    def append(self, other: Union["ParticleDataset", Any]):
        self._check_appendability(other)

        self.iteration.append(other.iteration)
        self.time.append(other.time)
        self.num_particles.append(other.num_particles)

        for q in self.quantities:
            getattr(self, q).append(getattr(other, q))


DatasetTypes = TypeVar("DatasetTypes", GridDataset, ParticleDataset)


@attr.s
class DatasetCollection:
    root_path: Path = attr.ib(converter=Path, validator=location_exists)
    _container_set: Set[DatasetTypes] = set([GridDataset, ParticleDataset])
    store: Dict[str, DatasetTypes] = attr.ib(factory=dict)

    def info(self, full: bool = False):  # pragma: no cover
        return self.__repr__()

    @property
    def datasets(self):
        return np.array([k for k in self.store.keys()], dtype=str)

    def _append_datasetcollection(self, obj):
        self.store.update(obj.store)

    def _append_file(self, obj):
        for container in self._container_set:
            try:
                dataset = container(obj)
                break
            except NataInvalidContainer:
                continue
        else:
            # not possible to append the file -> not a valid container found
            return

        if dataset.name in self.store:
            existing_ds = self.store[dataset.name]

            if existing_ds.appendable:
                existing_ds.append(dataset)
            else:
                raise ValueError(
                    f"Dataset '{existing_ds.name}' is not appendable!"
                )
        else:
            self.store[dataset.name] = dataset

    def append(self, obj: Union[str, Path, "DatasetCollection"]) -> None:
        """Takes a path to a diagnostic and appends it to the collection."""
        if isinstance(obj, DatasetCollection):
            self._append_datasetcollection(obj)

        elif isinstance(obj, (str, Path)):
            self._append_file(obj)

        else:
            raise ValueError(
                f"Can not append object of type '{type(obj)}' to collection"
            )

    def __getitem__(self, key):
        return self.store[key]

    @classmethod
    def register_plugin(cls, plugin_name, plugin):
        setattr(cls, plugin_name, plugin)
