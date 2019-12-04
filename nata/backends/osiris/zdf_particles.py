from pathlib import Path

from nata.utils.zdf import read, info
import numpy as np

from nata.backends import BaseParticles
from nata.containers import ParticleDataset, register_backend


@register_backend(ParticleDataset)
class Osiris_zdf_ParticleFile(BaseParticles):
    name = "osiris_zdf_particles"

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
    def short_name(self) -> str:
        z_info = info(str(self.location))
        return z_info.particles.name

    @property
    def num_particles(self) -> int:
        z_info = info(str(self.location))
        return z_info.particles.nparts

    @property
    def has_tags(self):
        z_info = info(str(self.location))
        return "tag" in z_info.particles.quants

    @property
    def tags(self):
        if not self.has_tags:
            raise AttributeError(
                f'The file "{self.location}" does not include tags!'
            )

        z_info = info(str(self.location))
        tags = z_info.particles.quants["tag"]
        return set((node, tag) for node, tag in tags)

    @property
    def quantities_list(self):
        z_info = info(str(self.location))
        quantities = []
        for key in z_info.particles.quants:
            if key == "tag":
                continue
            quantities.append(key)
                
        return np.array(quantities)

    @property
    def dataset(self):
        (z_data, z_info) = read(str(self.location))

        # create a structured array
        dset = np.zeros(self.num_particles, dtype=self.dtype)

        # fill the array
        for quant in self.quantities_list:
            dset[quant] = z_data[quant]

        return dset

    @property
    def quantities_long_names(self):
        z_info = info(str(self.location))
        names = {}
        for quant in self.quantities_list:
            names[quant] = z_info.particles.labels[quant]
        return names

    @property
    def quantities_units(self):
        z_info = info(str(self.location))
        units = dict()
        for quant in self.quantities_list:
            units[quant] = z_info.particles.units[quant]

        return units

    @property
    def dtype(self):
        (z_data, z_info) = read(str(self.location))
        fields = []
        for quant in self.quantities_list:
            fields.append((quant, z_data[quant].dtype))
        return np.dtype(fields)

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

