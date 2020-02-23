# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union

import numpy as np

from nata.containers import GridDataset
from nata.containers import register_backend
from nata.utils.zdf import info
from nata.utils.zdf import read

from ..grid import GridBackend


@register_backend(GridDataset)
class Osiris_zdf_GridFile(GridBackend):
    name = "osiris_zdf_grid"

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

    @property
    def _dset_name(self) -> str:
        z_info = info(str(self.location))
        label = z_info.grid.label
        return z_info.grid.name or self.clean(label)

    def get_data(self, indexing):
        # TODO: apply indexing here
        (z_data, z_info) = read(str(self.location))
        return z_data.transpose()

    @property
    def dataset_name(self) -> str:
        z_info = info(str(self.location))
        label = z_info.grid.label
        return z_info.grid.name or self.clean(label)

    @property
    def dataset_label(self) -> str:
        z_info = info(str(self.location))
        return z_info.grid.label

    @property
    def dim(self):
        z_info = info(str(self.location))
        return z_info.grid.ndims

    @property
    def shape(self):
        z_info = info(str(self.location))
        return tuple(z_info.grid.nx)

    @property
    def dtype(self):
        (z_data, z_info) = read(str(self.location))
        return z_data.dtype

    @property
    def dataset_unit(self):
        z_info = info(str(self.location))
        return z_info.grid.units

    @property
    def axes_min(self):
        min_values = []
        z_info = info(str(self.location))
        for axis in z_info.grid.axis:
            min_values.append(axis.min)
        return np.array(min_values)

    @property
    def axes_max(self):
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
            label = axis.label
            names.append(axis.name or self.clean(label))
        return np.array(names)

    @property
    def axes_labels(self):
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
