# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union

import h5py as h5
import numpy as np

from nata.containers import ParticleDataset
from nata.containers import register_backend

from ..particles import ParticleBackend


@register_backend(ParticleDataset)
class Osiris_Hdf5_ParticleFile(ParticleBackend):
    name = "osiris_hdf5_particles"

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
            if "TYPE" in f.attrs:
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

    @property
    def has_tags(self):
        with h5.File(self.location, mode="r") as fp:
            return "tag" in fp

    @property
    def tags(self):
        if not self.has_tags:
            raise AttributeError(
                f'The file "{self.location}" does not include tags!'
            )

        with h5.File(self.location, mode="r") as fp:
            tag_dset = fp["tag"]
            tags = np.zeros(tag_dset.shape, dtype=tag_dset.dtype)
            tag_dset.read_direct(tags)

        return set((node, tag) for node, tag in tags)

    # TODO: allow indexing
    def get_data(self, indexing=None, fields=None):
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
    def quantities(self):
        quantities = []

        with h5.File(self.location, mode="r") as fp:
            for key, item in fp.items():
                if key == "tag":
                    continue
                if isinstance(item, h5.Dataset):
                    quantities.append(key)

        return np.array(quantities)

    @property
    def quantity_labels(self):
        names = []
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantities:
                names.append(fp[quant].attrs["LONG_NAME"].astype(str)[0])
        return names

    @property
    def quantity_units(self):
        units = []
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantities:
                units.append(fp[quant].attrs["UNITS"].astype(str)[0])

        return units

    @property
    def dtype(self):
        fields = []
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantities:
                fields.append((quant, fp[quant].dtype))
        return np.dtype(fields)

    @property
    def iteration(self):
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["ITER"][0]

    @property
    def time_step(self):
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["TIME"][0]

    @property
    def time_unit(self):
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["TIME UNITS"].astype(str)[0]
