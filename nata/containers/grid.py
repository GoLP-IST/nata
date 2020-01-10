from pathlib import Path
from typing import Dict, Set, ValuesView, Optional
from copy import copy

import attr
import numpy as np

from nata.containers.base import BaseDataset
from nata.backends.grid import BaseGrid
from nata.utils.exceptions import NataInvalidContainer
from nata.utils import props_as_arr
from nata.utils.info_printer import PrettyPrinter


def _extract_grid_keys(key, spatial=True):
    if spatial:
        return np.index_exp[key[0]], np.index_exp[key[1:]]
    else:
        return np.index_exp[key[0]]


@attr.s
class GridDataset(BaseDataset):
    # GridDataset specific things
    _backends: Set[BaseGrid] = set()
    backend: BaseGrid.name = attr.ib(init=False)
    appendable = True
    num_entries: int = attr.ib(init=False, default=1)  # TODO: not yet used

    # data storage
    grid_obj: ValuesView = attr.ib(init=False, repr=False)
    store: Dict[int, BaseGrid] = attr.ib(init=False, factory=dict, repr=False)

    # information about object - grid object
    name: str = attr.ib(init=False)
    label: str = attr.ib(init=False)
    dimension: int = attr.ib(init=False)
    shape: tuple = attr.ib(init=False)
    unit: str = attr.ib(init=False)
    _axes_min: Dict[int, np.ndarray] = attr.ib(
        init=False, factory=dict, repr=False
    )
    _axes_max: Dict[int, np.ndarray] = attr.ib(
        init=False, factory=dict, repr=False
    )
    axes_names: np.ndarray = attr.ib(init=False)
    axes_labels: np.ndarray = attr.ib(init=False)
    axes_units: np.ndarray = attr.ib(init=False)
    time_unit: np.ndarray = attr.ib(init=False)

    @property
    def axes_min(self):
        if len(self._axes_min) == 1:
            return next(iter(self._axes_min.values()))

        return np.array([val for val in self._axes_min.values()])

    @property
    def axes_max(self):
        if len(self._axes_max) == 1:
            return next(iter(self._axes_max.values()))

        return np.array([val for val in self._axes_max.values()])

    @property
    def num_entries(self):
        return len(self.store)

    @property
    def iterations(self):
        if len(self.grid_obj) == 1:
            return np.array(getattr(next(iter(self.grid_obj)), "iteration"))
        return props_as_arr(self.grid_obj, "iteration", int)

    @property
    def time(self):
        if len(self.grid_obj) == 1:
            return np.array(getattr(next(iter(self.grid_obj)), "time_step"))
        return props_as_arr(self.grid_obj, "time_step", float)

    @property
    def data(self) -> np.ndarray:
        if len(self.store) == 1:
            grid_obj = next(iter(self.store.values()))
            return grid_obj.dataset

        shape = (len(self.store),) + self.shape
        d = np.zeros(shape)
        for i, iteration in enumerate(self.iterations):
            d[i] = self.store[iteration].dataset
        return d

    @property
    def backend_name(self) -> str:
        backend: BaseGrid = next(iter(self.store.values()))
        return backend.name

    # using post init from attr.s -> obj will always represent
    def __attrs_post_init__(self):
        # select backend - first matching is used
        for backend in self._backends:
            if backend.is_valid_backend(self.location):
                grid_obj = backend(self.location)
                break
        else:
            raise NataInvalidContainer

        self.backend = grid_obj.name
        self.name = grid_obj.short_name
        self.label = grid_obj.long_name
        self.dimension = grid_obj.dim
        self.shape = grid_obj.shape
        self.unit = grid_obj.dataset_unit
        self._axes_min[grid_obj.iteration] = grid_obj.axis_min
        self._axes_max[grid_obj.iteration] = grid_obj.axis_max
        self.axes_names = grid_obj.axes_names
        self.axes_labels = grid_obj.axes_long_names
        self.axes_units = grid_obj.axes_units
        self.time_unit = grid_obj.time_unit

        self.location = grid_obj.location.parent
        self.store[grid_obj.iteration] = grid_obj
        self.grid_obj = self.store.values()

    def __getitem__(self, key):
        if isinstance(key, slice) or isinstance(key, int):
            new = copy(self)
            new._reduce_store(new.iterations[key])
            new._update_grid()
            new._reduce_axes()
            return new

        if isinstance(key, tuple) and len(key) != (self.dimension + 1):
            raise KeyError(
                f"Insufficient indices for a {self.dimension}d grid! "
                + "Required for temporal and spatial selection!"
            )

        temporal_keys, spatial_keys = _extract_grid_keys(key)

        new = copy(self)
        new._reduce_store(new.iterations[temporal_keys])
        new._update_grid(spatial_keys)
        new._reduce_axes()
        new._update_props()

        return new

    def _reduce_store(self, to_keep: np.ndarray):
        # if single item - is passed
        if not to_keep.shape:
            to_keep = (to_keep,)

        self.store = {
            key: val for key, val in self.store.items() if key in to_keep
        }

    def _update_grid(self, spatial_keys=None):
        # check if copying grid object and updating selection is required
        if spatial_keys:
            if any([s != slice(None) for s in spatial_keys]):
                for key, val in self.store.items():
                    self.store[key] = copy(val)
                    self.store[key].selection = spatial_keys

        self.grid_obj = self.store.values()

    def _reduce_axes(self):
        self._axes_min = {key: val.axis_min for key, val in self.store.items()}
        self._axes_max = {key: val.axis_max for key, val in self.store.items()}

    def _update_props(self):
        # currently only the shape has to be updated -> read the first entry
        self.shape = getattr(next(iter(self.grid_obj)), "shape")

    def info(
        self,
        printer: Optional[PrettyPrinter] = None,
        root_path: Optional[Path] = None,
    ):  # pragma: no cover
        requires_flush = False

        if printer is None:
            printer = PrettyPrinter(header=f"Dataset '{self.name}'")
            requires_flush = True
        else:
            printer.add_line(f'* Dateset: "{self.name}" *')
            printer.add_line("-" * (15 + len(self.name)))
        if root_path is None:
            # global root and prints absolute path
            root_path = Path.cwd()

        printer.indent()
        printer.add_line(f'full name: "{self.label}"')
        printer.add_line(
            f"location: " + f"{self.location.relative_to(root_path)}"
        )
        printer.add_line(f'backend: "{self.backend_name}"')
        printer.add_line(f"entries: {self.num_entries}")
        if (
            not isinstance(self.iterations, (np.int, np.int32, np.int64, int))
            and len(self.iterations) > 4
        ):
            printer.add_line(
                f"iterations: ["
                + f"{self.iterations[0]} "
                + f"{self.iterations[1]} "
                + f"... {self.iterations[-1]} ] "
            )
            printer.add_line(
                f"time_steps: ["
                + f"{self.time[0]} "
                + f"{self.time[1]} "
                + f"... {self.time[-1]} ] "
            )
        else:
            printer.add_line(f"iterations: {self.iterations}")
            printer.add_line(f"time steps: {self.time}")

        printer.add_line(f"dim: {self.dimension}")
        printer.add_line(f"shape: {self.shape}")
        printer.add_line(f"axes: {self.axes_names}")
        printer.undent()

        if requires_flush:
            printer.flush()

    def append(self, other: "GridDataset"):
        # TODO: Find a better way to check both types if they are mergable
        if not isinstance(other, GridDataset):
            raise ValueError(
                "Can not append something of different container type!"
            )
        if not self.name == other.name:
            raise ValueError(
                "Mismatch in names for container."
                + "Only equally named containers can be appended!"
            )
        if not self.backend_name == other.backend_name:
            raise ValueError("Mixed backends are not allowed!")

        self.store.update(other.store)
        self._axes_min.update(other._axes_min)
        self._axes_max.update(other._axes_max)
