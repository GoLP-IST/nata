from pathlib import Path

import h5py as h5
import numpy as np

from nata.backends import BaseParticles
from nata.containers import ParticleDataset, register_backend


@register_backend(ParticleDataset)
class Osiris_Hdf5_ParticleFile(BaseParticles):
    name = "osiris_hdf5_particles"

    @staticmethod
    def is_valid_backend(file_path: Path) -> bool:
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
    def short_name(self) -> str:
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

    @property
    def quantities_list(self):
        quantities = []

        with h5.File(self.location, mode="r") as fp:
            for key, item in fp.items():
                if key == "tag":
                    continue
                if isinstance(item, h5.Dataset):
                    quantities.append(key)

        return np.array(quantities)

    @property
    def dataset(self):
        with h5.File(self.location, mode="r") as fp:
            # create a structured array
            dset = np.zeros(self.num_particles, dtype=self.dtype)

            # fill the array
            for quant in self.quantities_list:
                dset[quant] = fp[quant]

        return dset

    @property
    def quantities_long_names(self):
        names = {}
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantities_list:
                names[quant] = fp[quant].attrs["LONG_NAME"].astype(str)[0]
        return names

    @property
    def quantities_units(self):
        units = dict()
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantities_list:
                units[quant] = fp[quant].attrs["UNITS"].astype(str)[0]

        return units

    @property
    def dtype(self):
        fields = []
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantities_list:
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
