from typing import Set, ValuesView, Dict, Optional
from pathlib import Path
from copy import copy

import attr
import numpy as np

from nata.containers import BaseDataset
from nata.backends import BaseParticles
from nata.utils import props_as_arr
from nata.utils.exceptions import NataInvalidContainer
from nata.utils.info_printer import PrettyPrinter


@attr.s
class ParticleDataset(BaseDataset):
    _backends: Set[BaseParticles] = set()
    backend: BaseParticles.name = attr.ib(init=False)
    appendable = True

    # data storage
    prt_objs: ValuesView = attr.ib(init=False, repr=False)
    store: Dict[int, BaseParticles] = attr.ib(
        init=False, factory=dict, repr=False
    )

    # information about the object - particle object
    name: str = attr.ib(init=False)
    quantities: np.ndarray = attr.ib(init=False)
    quantities_labels: Dict[str, str] = attr.ib(init=False)
    quantities_units: Dict[str, str] = attr.ib(init=False)
    tagged: bool = attr.ib(init=False)
    _dtype: np.dtype = attr.ib(init=False)
    _num_particles: Dict[int, int] = attr.ib(init=False, factory=dict)

    time_unit: np.ndarray = attr.ib(init=False)

    @property
    def data(self) -> np.ndarray:
        if len(self.store) == 1:
            prt_obj = next(iter(self.store.values()))
            return prt_obj.dataset

        shape = (len(self.store), max(self._num_particles.values()))
        d = np.ma.zeros(shape, dtype=self._dtype)
        d.mask = np.zeros(shape, dtype=bool)

        for i, iteration in enumerate(self.iterations):
            num_prt = self._num_particles[iteration]
            d[i, :num_prt] = self.store[iteration].dataset
            d.mask[i, num_prt:] = True
        return d

    @property
    def iterations(self):
        return props_as_arr(self.prt_objs, "iteration", int)

    @property
    def time(self):
        return props_as_arr(self.prt_objs, "time_step", float)

    @property
    def backend_name(self) -> str:
        backend = next(iter(self.store.values()))
        return backend.name

    def __attrs_post_init__(self):
        for backend in self._backends:
            if backend.is_valid_backend(self.location):
                prt_obj = backend(self.location)
                break
        else:
            raise NataInvalidContainer

        self.backend = prt_obj.name
        self.name = prt_obj.short_name
        self.quantities = prt_obj.quantities_list
        self.quantities_labels = prt_obj.quantities_long_names
        self.quantities_units = prt_obj.quantities_units
        self._num_particles[prt_obj.iteration] = prt_obj.num_particles
        self._dtype = prt_obj.dtype
        self.tagged = prt_obj.has_tags
        self.time_unit = prt_obj.time_unit

        self.location = prt_obj.location.parent
        self.store[prt_obj.iteration] = prt_obj
        self.prt_objs = self.store.values()

    def __getitem__(self, key):
        if isinstance(key, slice) or isinstance(key, int):
            new = copy(self)
            new._reduce_store(new.iterations[key])
            new.prt_objs = new.store.values()
            return new

        raise NotImplementedError(
            "Currently only temporal slices are supported!"
        )

    def _reduce_store(self, to_keep: np.ndarray):
        # if single item - is passed
        if not to_keep.shape:
            to_keep = (to_keep,)

        self.store = {
            key: val for key, val in self.store.items() if key in to_keep
        }

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
        printer.add_line(f"location: {self.location.relative_to(root_path)}")
        printer.add_line(f"backend: {self.backend_name}")
        printer.add_line(
            f"num. particles (max): {max(self._num_particles.values())}"
        )
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

        printer.add_line(f"quantaties: {self.quantities}")
        printer.undent()

        if requires_flush:
            printer.flush()

    def append(self, other: "ParticleDataset"):
        # TODO: Find a better way to check both types if they are mergable
        if not isinstance(other, ParticleDataset):
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
        self._num_particles.update(other._num_particles)
