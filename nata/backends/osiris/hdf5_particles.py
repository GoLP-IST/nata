# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import h5py as h5
import numpy as np

from nata.containers import ParticleDataset
from nata.types import ParticleBackendType
from nata.utils.container_tools import register_backend


@register_backend(ParticleDataset)
class Osiris_Hdf5_ParticleFile(ParticleBackendType):
    name = "osiris_hdf5_particles"
    location: Optional[Path] = None

    def __init__(self, location=Union[str, Path]) -> None:
        self.location = (
            location if isinstance(location, Path) else Path(location)
        )

    @staticmethod
    def is_valid_backend(path: Union[Path, str]) -> bool:
        if isinstance(path, str):
            path = Path(path)

        if not isinstance(path, Path):
            return False

        if not path.is_file():
            return False

        if not path.suffix == ".h5":
            return False

        if not h5.is_hdf5(path):
            return False

        with h5.File(path, mode="r") as f:
            if ("TYPE" in f.attrs) and ("LABELS" not in f.attrs):
                type_ = f.attrs["TYPE"].astype(str)[0]

                if type_ == "particles":
                    return True

        return False

    @property
    def dataset_name(self) -> str:
        with h5.File(self.location, mode="r") as fp:
            dataset_name = fp.attrs["NAME"].astype(str)[0]
        return dataset_name

    @property
    def num_particles(self) -> int:
        with h5.File(self.location, mode="r") as fp:
            num_particles: int = fp["q"].shape[0]
        return num_particles

    def get_data(self, indexing=None, fields=None) -> np.ndarray:
        with h5.File(self.location, mode="r") as fp:
            if fields is None:
                # create a structured array
                dset = np.empty(self.num_particles, dtype=self.dtype)

                # fill the array
                for quant in self.quantity_names:
                    dset[quant] = fp[quant]
            else:
                if indexing is None:
                    dset = fp[fields][:]
                else:
                    dset = fp[fields][indexing]

        return dset

    @property
    def quantity_names(self) -> List[str]:
        quantities = []

        with h5.File(self.location, mode="r") as fp:
            for key, item in fp.items():
                if key == "tag":
                    continue
                if isinstance(item, h5.Dataset):
                    quantities.append(key)

        return quantities

    @property
    def quantity_labels(self) -> List[str]:
        names = []
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantity_names:
                name: str = fp[quant].attrs["LONG_NAME"].astype(str)[0]
                names.append(name)
        return names

    @property
    def quantity_units(self) -> List[str]:
        units = []
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantity_names:
                units.append(fp[quant].attrs["UNITS"].astype(str)[0])

        return units

    @property
    def dtype(self) -> np.dtype:
        fields = []
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantity_names:
                fields.append((quant, fp[quant].dtype))
        return np.dtype(fields)

    @property
    def iteration(self) -> int:
        with h5.File(self.location, mode="r") as fp:
            iteration = fp.attrs["ITER"][0]
        return iteration

    @property
    def time_step(self) -> float:
        with h5.File(self.location, mode="r") as fp:
            time_step = fp.attrs["TIME"][0]
        return time_step

    @property
    def time_unit(self) -> str:
        with h5.File(self.location, mode="r") as fp:
            time_unit = fp.attrs["TIME UNITS"].astype(str)[0]
        return time_unit
