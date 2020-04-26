# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import h5py as h5
import numpy as np

from nata.containers import ParticleDataset
from nata.utils.container_tools import register_backend


@register_backend(ParticleDataset)
class Osiris_Dev_Hdf5_ParticleFile:
    name = "osiris_dev_hdf5_particles"
    location: Optional[Path] = None

    def __init__(self, location=Union[str, Path]) -> None:
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

        if not file_path.suffix == ".h5":
            return False

        if not h5.is_hdf5(file_path):
            return False

        with h5.File(file_path, mode="r") as f:
            if ("TYPE" in f.attrs) and ("LABELS" in f.attrs):
                type_ = f.attrs["TYPE"].astype(str)[0]

                if type_ == "particles":
                    return True

        return False

    @property
    def dataset_name(self) -> str:
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["NAME"].astype(str)[0]

    @property
    def num_particles(self) -> int:
        with h5.File(self.location, mode="r") as fp:
            return fp["q"].shape[0]

    def get_data(self, indexing=None, fields=None) -> np.ndarray:
        with h5.File(self.location, mode="r") as fp:
            if fields is None:
                # create a structured array
                dset = np.empty(self.num_particles, dtype=self.dtype)

                # fill the array
                for quant in self.quantities:
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
        ordered_quants = self.quantities
        labels = []

        with h5.File(self.location, mode="r") as fp:
            unordered_labels = fp.attrs["LABELS"].astype(str)
            unordered_quants = fp.attrs["QUANTS"].astype(str)

            order = []
            for quant in unordered_quants:
                order.append(ordered_quants.index(quant))

            labels = [unordered_labels[i] for i in order]

        return labels

    @property
    def quantity_units(self) -> List[str]:
        ordered_quants = self.quantities
        units = []

        with h5.File(self.location, mode="r") as fp:
            unordered_units = fp.attrs["UNITS"].astype(str)
            unordered_quants = fp.attrs["QUANTS"].astype(str)

            order = []
            for quant in unordered_quants:
                order.append(ordered_quants.index(quant))

            units = [unordered_units[i] for i in order]

        return units

    @property
    def dtype(self) -> np.dtype:
        fields = []
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantities:
                fields.append((quant, fp[quant].dtype))
        return np.dtype(fields)

    @property
    def iteration(self) -> int:
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["ITER"][0]

    @property
    def time_step(self) -> float:
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["TIME"][0]

    @property
    def time_unit(self) -> str:
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["TIME UNITS"].astype(str)[0]
