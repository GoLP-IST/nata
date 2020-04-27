# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from nata.containers import GridDataset
from nata.containers import ParticleDataset
from nata.utils.cached_property import cached_property
from nata.utils.container_tools import register_backend
from nata.utils.zdf import info
from nata.utils.zdf import read


@register_backend(GridDataset)
class Osiris_zdf_GridFile:
    name = "osiris_dev_grid_zdf"
    location: Optional[Path] = None

    def __init__(self, location: Union[str, Path]) -> None:
        self.location = (
            location if isinstance(location, Path) else Path(location)
        )

    @staticmethod
    def is_valid_backend(file_path: Union[Path, str]) -> bool:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not isinstance(file_path, Path):
            return False

        if not file_path.is_file():
            return False

        if not file_path.suffix == ".zdf":
            return False

        z_info = info(str(file_path))
        if hasattr(z_info, "type"):
            if z_info.type == "grid":
                return True

        return False

    @cached_property
    def _dset_name(self) -> str:
        logging.info(f"Accessing '{self.location}' for '_dset_name'")
        z_info = info(str(self.location))
        label = z_info.grid.label
        return z_info.grid.name or self.clean(label)

    def get_data(self=None, indexing=None):
        # TODO: apply indexing here
        (z_data, z_info) = read(str(self.location))
        return z_data.transpose()

    @cached_property
    def dataset_name(self) -> str:
        logging.info(f"Accessing '{self.location}' for 'dataset_name'")
        z_info = info(str(self.location))
        label = z_info.grid.label
        return z_info.grid.name or self.clean(label)

    @cached_property
    def dataset_label(self) -> str:
        logging.info(f"Accessing '{self.location}' for 'dataset_label'")
        z_info = info(str(self.location))
        return z_info.grid.label

    @cached_property
    def ndim(self):
        logging.info(f"Accessing '{self.location}' for 'ndim'")
        z_info = info(str(self.location))
        return z_info.grid.ndims

    @cached_property
    def shape(self):
        logging.info(f"Accessing '{self.location}' for 'shape'")
        z_info = info(str(self.location))
        nx = []
        for n in z_info.grid.nx:
            nx.append(n.item())
        return tuple(nx)

    @cached_property
    def dtype(self):
        logging.info(f"Accessing '{self.location}' for 'dtype'")
        (z_data, z_info) = read(str(self.location))
        return z_data.dtype

    @cached_property
    def dataset_unit(self):
        logging.info(f"Accessing '{self.location}' for 'dataset_unit'")
        z_info = info(str(self.location))
        return z_info.grid.units

    @cached_property
    def axes_min(self):
        logging.info(f"Accessing '{self.location}' for 'axes_min'")
        min_values = []
        z_info = info(str(self.location))
        for axis in z_info.grid.axis:
            min_values.append(axis.min)
        return np.array(min_values)

    @cached_property
    def axes_max(self):
        logging.info(f"Accessing '{self.location}' for 'axes_max'")
        max_values = []
        z_info = info(str(self.location))
        for axis in z_info.grid.axis:
            max_values.append(axis.max)
        return np.array(max_values)

    @cached_property
    def axes_names(self):
        logging.info(f"Accessing '{self.location}' for 'axes_names'")
        names = []
        z_info = info(str(self.location))
        for axis in z_info.grid.axis:
            label = axis.label
            names.append(axis.name or self.clean(label))
        return np.array(names)

    @cached_property
    def axes_labels(self):
        logging.info(f"Accessing '{self.location}' for 'axes_labels'")
        long_names = []
        z_info = info(str(self.location))
        for axis in z_info.grid.axis:
            long_names.append(axis.label)
        return np.array(long_names)

    @cached_property
    def axes_units(self):
        logging.info(f"Accessing '{self.location}' for 'axes_units'")
        units = []
        z_info = info(str(self.location))
        for axis in z_info.grid.axis:
            units.append(axis.units)
        return np.array(units)

    @cached_property
    def iteration(self):
        logging.info(f"Accessing '{self.location}' for 'iteration'")
        z_info = info(str(self.location))
        return z_info.iteration.n

    @cached_property
    def time_step(self):
        logging.info(f"Accessing '{self.location}' for 'time_step'")
        z_info = info(str(self.location))
        return z_info.iteration.t

    @cached_property
    def time_unit(self):
        logging.info(f"Accessing '{self.location}' for 'time_unit'")
        z_info = info(str(self.location))
        return z_info.iteration.tunits

    def clean(self, name):
        return (
            name.lower()
            .replace("_", "")
            .replace("^", "")
            .replace("{", "")
            .replace("}", "")
            .replace("\\", "")
            .replace(" ", "_")
        )


@register_backend(ParticleDataset)
class Osiris_zdf_ParticleFile:
    name = "osiris_dev_particles_zdf"
    location: Optional[Path] = None

    def __init__(self, location=Union[str, Path]) -> None:
        self.location = (
            location if isinstance(location, Path) else Path(location)
        )

    @staticmethod
    def is_valid_backend(file_path: Path) -> bool:
        if not file_path.is_file():
            return False

        if not file_path.suffix == ".zdf":
            return False

        z_info = info(str(file_path))
        if hasattr(z_info, "type"):
            if z_info.type == "particles":
                return True

        return False

    @cached_property
    def dataset_name(self) -> str:
        logging.info(f"Accessing '{self.location}' for 'dataset_name'")
        z_info = info(str(self.location))
        return z_info.particles.name

    @cached_property
    def num_particles(self) -> int:
        logging.info(f"Accessing '{self.location}' for 'num_particles'")
        z_info = info(str(self.location))
        return z_info.particles.nparts

    def get_data(self, indexing=None, fields=None) -> np.ndarray:
        (z_data, z_info) = read(str(self.location))

        # create a structured array
        dset = np.zeros(self.num_particles, dtype=self.dtype)

        # fill the array
        for quant in self.quantity_names:
            dset[quant] = z_data[quant]

        return dset

    @cached_property
    def quantity_names(self) -> List[str]:
        logging.info(f"Accessing '{self.location}' for 'quantity_names'")
        z_info = info(str(self.location))
        quantities = []
        for key in z_info.particles.quants:
            if key == "tag":
                continue
            quantities.append(key)

        return quantities

    @cached_property
    def quantity_labels(self) -> List[str]:
        logging.info(f"Accessing '{self.location}' for 'quantity_labels'")
        z_info = info(str(self.location))
        names = []
        for quant in self.quantity_names:
            names.append(z_info.particles.labels[quant])
        return names

    @cached_property
    def quantity_units(self) -> List[str]:
        logging.info(f"Accessing '{self.location}' for 'quantity_units'")
        z_info = info(str(self.location))
        units = []
        for quant in self.quantity_names:
            units.append(z_info.particles.units[quant])

        return units

    @cached_property
    def dtype(self) -> np.dtype:
        logging.info(f"Accessing '{self.location}' for 'dtype'")
        (z_data, z_info) = read(str(self.location))
        fields = []
        for quant in self.quantity_names:
            fields.append((quant, z_data[quant].dtype))
        return np.dtype(fields)

    @cached_property
    def iteration(self) -> int:
        logging.info(f"Accessing '{self.location}' for 'iteration'")
        z_info = info(str(self.location))
        return z_info.iteration.n

    @cached_property
    def time_step(self) -> float:
        logging.info(f"Accessing '{self.location}' for 'time_step'")
        z_info = info(str(self.location))
        return z_info.iteration.t

    @cached_property
    def time_unit(self) -> str:
        logging.info(f"Accessing '{self.location}' for 'time_unit'")
        z_info = info(str(self.location))
        return z_info.iteration.tunits
