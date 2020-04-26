# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from nata.containers import ParticleDataset
from nata.utils.container_tools import register_backend
from nata.utils.zdf import info
from nata.utils.zdf import read


@register_backend(ParticleDataset)
class Osiris_zdf_ParticleFile:
    name = "osiris_zdf_particles"
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

    @property
    def dataset_name(self) -> str:
        z_info = info(str(self.location))
        return z_info.particles.name

    @property
    def num_particles(self) -> int:
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

    @property
    def quantity_names(self) -> List[str]:
        z_info = info(str(self.location))
        quantities = []
        for key in z_info.particles.quants:
            if key == "tag":
                continue
            quantities.append(key)

        return quantities

    @property
    def quantity_labels(self) -> List[str]:
        z_info = info(str(self.location))
        names = []
        for quant in self.quantity_names:
            names.append(z_info.particles.labels[quant])
        return names

    @property
    def quantity_units(self) -> List[str]:
        z_info = info(str(self.location))
        units = []
        for quant in self.quantity_names:
            units.append(z_info.particles.units[quant])

        return units

    @property
    def dtype(self) -> np.dtype:
        (z_data, z_info) = read(str(self.location))
        fields = []
        for quant in self.quantity_names:
            fields.append((quant, z_data[quant].dtype))
        return np.dtype(fields)

    @property
    def iteration(self) -> int:
        z_info = info(str(self.location))
        return z_info.iteration.n

    @property
    def time_step(self) -> float:
        z_info = info(str(self.location))
        return z_info.iteration.t

    @property
    def time_unit(self) -> str:
        z_info = info(str(self.location))
        return z_info.iteration.tunits
