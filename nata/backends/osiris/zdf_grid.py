from pathlib import Path
from typing import Union

from nata.utils.zdf import read, info
import numpy as np

from nata.backends import BaseGrid
from nata.containers import GridDataset, register_backend


@register_backend(GridDataset)
class Osiris_zdf_GridFile(BaseGrid):
    name = "osiris_zdf_grid"

    @staticmethod
    def is_valid_backend(file_path: Path) -> bool:
        if not file_path.is_file():
            return False

        if not file_path.suffix == ".zdf":
            return False

        z_info = info(str(file_path))
        if hasattr(z_info, "type"):
            if z_info.type == "grid":
                return True

        return False

    @property
    def short_name(self) -> str:
        z_info = info(str(self.location))
        return z_info.grid.name

    @property
    def long_name(self) -> str:
        z_info = info(str(self.location))
        return z_info.grid.label

    @property
    def dataset(self):
        (z_data, z_info) = read(str(self.location))
        return z_data.transpose()

    @property
    def dim(self):
        z_info = info(str(self.location))
        return z_info.grid.ndims.astype(int)

    @property
    def shape(self):
        z_info = info(str(self.location))
        return tuple(z_info.grid.nx.astype(int))

    @property
    def dtype(self):
        (z_data, z_info) = read(str(self.location))
        return z_data.dtype

    @property
    def dataset_unit(self):
        z_info = info(str(self.location))
        return z_info.grid.units

    @property
    def axis_min(self):
        min_values = []
        z_info = info(str(self.location))
        for axis in z_info.grid.axis:
            min_values.append(axis.min)
        return np.array(min_values)

    @property
    def axis_max(self):
        max_values = []
        z_info = info(str(self.location))
        for axis in z_info.grid.axis:
            max_values.append(axis.max)
        return np.array(max_values)

    @property
    def axes_names(self):
        names = []
        z_info = info(str(self.location))
        for axis in z_info.grid.axis:
            names.append(axis.name)
        return np.array(names)

    @property
    def axes_long_names(self):
        long_names = []
        z_info = info(str(self.location))
        for axis in z_info.grid.axis:
            long_names.append(axis.label)
        return np.array(long_names)

    @property
    def axes_units(self):
        units = []
        z_info = info(str(self.location))
        for axis in z_info.grid.axis:
            units.append(axis.units)
        return np.array(units)

    @property
    def iteration(self):
        z_info = info(str(self.location))
        return z_info.iteration.n

    @property
    def time_step(self):
        z_info = info(str(self.location))
        return z_info.iteration.t

    @property
    def time_unit(self):
        z_info = info(str(self.location))
        return z_info.iteration.tunits
