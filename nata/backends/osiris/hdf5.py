# -*- coding: utf-8 -*-
from logging import info
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import h5py as h5
import numpy as np

from nata.containers import GridDataset
from nata.containers import ParticleDataset
from nata.types import FileLocation
from nata.utils.backends import sort_particle_quantities
from nata.utils.cached_property import cached_property
from nata.utils.container_tools import register_backend


@register_backend(GridDataset)
class Osiris_Hdf5_GridFile:
    name = "osiris_4.4.4_grid_hdf5"
    location: Optional[Path] = None

    def __init__(self, location: Union[str, Path]) -> None:
        self.location = (
            location if isinstance(location, Path) else Path(location)
        )

        info(f"Obtaining backend props for '{self.location}'")
        with h5.File(self.location, mode="r") as fp:
            short_name = fp.attrs["NAME"].astype(str)[0]
            if short_name in fp:
                self._dset_name = short_name
            else:
                name_ = short_name.split()[-1]
                self._dset_name = name_.replace("_", "")

            self._dataset_name = fp.attrs["NAME"].astype(str)[0]
            self._dataset_label = (
                fp[self._dset_name].attrs["LONG_NAME"].astype(str)[0]
            )
            self._ndim = fp[self._dset_name].ndim
            self._shape = fp[self._dset_name].shape[::-1]
            self._dtype = fp[self._dset_name].dtype
            self._units = fp[self._dset_name].attrs["UNITS"].astype(str)[0]

            min_, max_ = [], []
            axes_names, axes_labels, axes_units = [], [], []
            for ax in fp["AXIS"]:
                min_.append(fp["AXIS/" + ax][0])
                max_.append(fp["AXIS/" + ax][1])
                axes_names.append(fp["AXIS/" + ax].attrs["NAME"].astype(str)[0])
                axes_labels.append(
                    fp["AXIS/" + ax].attrs["LONG_NAME"].astype(str)[0]
                )
                axes_units.append(
                    fp["AXIS/" + ax].attrs["UNITS"].astype(str)[0]
                )

            self._min = np.array(min_)
            self._max = np.array(max_)
            self._axes_names = np.array(axes_names)
            self._axes_labels = np.array(axes_labels)
            self._axes_units = np.array(axes_units)

            self._iteration = fp.attrs["ITER"].astype(int)[0]
            self._time_step = fp.attrs["TIME"][0]
            self._time_unit = fp.attrs["TIME UNITS"].astype(str)[0]

    @staticmethod
    def is_valid_backend(path: FileLocation) -> bool:
        if isinstance(path, str):
            path = Path(path)

        if (
            not isinstance(path, Path)
            or not path.is_file()
            or not path.suffix == ".h5"
            or not h5.is_hdf5(path)
        ):
            return False

        with h5.File(path, mode="r") as f:
            if (
                ("NAME" in f.attrs)
                and ("TYPE" in f.attrs)
                and ("LABEL" not in f.attrs)
            ):
                type_ = f.attrs["TYPE"].astype(str)[0]
                # general naming
                name_ = f.attrs["NAME"].astype(str)[0]
                names = (name_,)
                # special case naming
                name_ = name_.split()[-1]
                name_ = name_.replace("_", "")

                names += (name_,)
                if (type_ == "grid") and any([name in f for name in names]):
                    return True

        return False

    def get_data(self, indexing=None):
        info(f"Reading data in '{self.location}'")
        with h5.File(self.location, mode="r") as fp:
            if indexing:
                dataset = fp[self._dset_name][indexing[::-1]]
            else:
                dataset = fp[self._dset_name][:]
        return dataset.transpose()

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_label(self) -> str:
        return self._dataset_label

    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def dataset_unit(self):
        return self._units

    @property
    def axes_min(self):
        return self._min

    @property
    def axes_max(self):
        return self._max

    @property
    def axes_names(self):
        return self._axes_names

    @property
    def axes_labels(self):
        return self._axes_labels

    @property
    def axes_units(self):
        return self._axes_units

    @property
    def iteration(self):
        return self._iteration

    @property
    def time_step(self):
        return self._time_step

    @property
    def time_unit(self):
        return self._time_unit


@register_backend(GridDataset)
class Osiris_Dev_Hdf5_GridFile:
    name = "osiris_dev_grid_hdf5"
    location: Optional[Path] = None

    def __init__(self, location: FileLocation) -> None:
        self.location = (
            location if isinstance(location, Path) else Path(location)
        )

        info(f"Obtaining backend props for '{self.location}'")
        with h5.File(self.location, mode="r") as fp:
            short_name = fp.attrs["NAME"].astype(str)[0]
            if short_name in fp:
                self._dset_name = short_name

            name_ = short_name.split()[-1]
            name_ = name_.replace("_", "")
            if name_ in fp:
                self._dset_name = name_

            self._dataset_name = fp.attrs["NAME"].astype(str)[0]
            self._dataset_label = fp.attrs["LABEL"].astype(str)[0]
            self._dataset_unit = fp.attrs["UNITS"].astype(str)[0]
            self._ndim = fp[self._dset_name].ndim
            self._shape = fp[self._dset_name].shape[::-1]
            self._dtype = fp[self._dset_name].dtype

            min_, max_ = [], []
            axes_names, axes_labels, axes_units = [], [], []
            for ax in fp["AXIS"]:
                min_.append(fp["AXIS/" + ax][0])
                max_.append(fp["AXIS/" + ax][1])
                axes_names.append(fp["AXIS/" + ax].attrs["NAME"].astype(str)[0])
                axes_labels.append(
                    fp["AXIS/" + ax].attrs["LONG_NAME"].astype(str)[0]
                )
                axes_units.append(
                    fp["AXIS/" + ax].attrs["UNITS"].astype(str)[0]
                )

            self._axes_min = np.array(min_)
            self._axes_max = np.array(max_)
            self._axes_names = np.array(axes_names)
            self._axes_labels = np.array(axes_labels)
            self._axes_units = np.array(axes_units)

            self._iteration = fp.attrs["ITER"].astype(int)[0]
            self._time_step = fp.attrs["TIME"][0]
            self._time_unit = fp.attrs["TIME UNITS"].astype(str)[0]

    @staticmethod
    def is_valid_backend(path: FileLocation) -> bool:
        if isinstance(path, str):
            path = Path(path)

        if (
            not isinstance(path, Path)
            or not path.is_file()
            or not path.suffix == ".h5"
            or not h5.is_hdf5(path)
        ):
            return False

        with h5.File(path, mode="r") as f:
            if (
                ("NAME" in f.attrs)
                and ("TYPE" in f.attrs)
                and ("LABEL" in f.attrs)
            ):
                type_ = f.attrs["TYPE"].astype(str)[0]
                # general naming
                name_ = f.attrs["NAME"].astype(str)[0]
                names = (name_,)
                # special case naming
                name_ = name_.split()[-1]
                name_ = name_.replace("_", "")

                names += (name_,)
                if (type_ == "grid") and any([name in f for name in names]):
                    return True

        return False

    def get_data(self, indexing=None):
        info(f"Reading data in '{self.location}'")
        with h5.File(self.location, mode="r") as fp:
            if indexing:
                dataset = fp[self._dset_name][indexing[::-1]]
            else:
                dataset = fp[self._dset_name][:]
        return dataset.transpose()

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_label(self) -> str:
        return self._dataset_label

    @property
    def dataset_unit(self):
        return self._dataset_unit

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def axes_min(self):
        return self._axes_min

    @property
    def axes_max(self):
        return self._axes_max

    @property
    def axes_names(self):
        return self._axes_names

    @property
    def axes_labels(self):
        return self._axes_labels

    @property
    def axes_units(self):
        return self._axes_units

    @property
    def iteration(self):
        return self._iteration

    @property
    def time_step(self):
        return self._time_step

    @property
    def time_unit(self):
        return self._time_unit


@register_backend(ParticleDataset)
class Osiris_Hdf5_ParticleFile:
    name = "osiris_4.4.4_particles_hdf5"
    location: Optional[Path] = None

    def __init__(self, location=Union[str, Path]) -> None:
        self.location = (
            location if isinstance(location, Path) else Path(location)
        )

    @staticmethod
    def is_valid_backend(path: FileLocation) -> bool:
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

    @cached_property
    def dataset_name(self) -> str:
        info(f"Accessing '{self.location}' for 'dataset_name'")
        with h5.File(self.location, mode="r") as fp:
            dataset_name = fp.attrs["NAME"].astype(str)[0]
        return dataset_name

    @cached_property
    def num_particles(self) -> int:
        info(f"Accessing '{self.location}' for 'num_particles'")
        with h5.File(self.location, mode="r") as fp:
            num_particles = fp["q"].shape[0] if fp["q"].shape else 0
        return num_particles

    def get_data(self, indexing=None, fields=None) -> np.ndarray:
        info(f"Reading data in '{self.location}'")
        with h5.File(self.location, mode="r") as fp:
            if fields is None:
                # create a structured array
                dset = np.empty(self.num_particles, dtype=self.dtype)

                # fill the array
                for quant in self.quantity_names:
                    dset[quant] = fp[quant]
            else:
                if indexing is None:
                    if fp[fields].shape:
                        dset = fp[fields][:]
                    else:
                        dset = np.empty(0, dtype=self.dtype[fields])
                else:
                    dset = fp[fields][indexing]

        return dset

    @cached_property
    def quantity_names(self) -> List[str]:
        info(f"Accessing '{self.location}' for 'quantity_names'")
        quantities = []

        with h5.File(self.location, mode="r") as fp:
            for key, item in fp.items():
                if key == "tag":
                    continue
                if isinstance(item, h5.Dataset):
                    quantities.append(key)

        return sort_particle_quantities(quantities, ["x", "p"])

    @cached_property
    def quantity_labels(self) -> List[str]:
        info(f"Accessing '{self.location}' for 'quantity_labels'")
        names = []
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantity_names:
                name: str = fp[quant].attrs["LONG_NAME"].astype(str)[0]
                names.append(name)
        return names

    @cached_property
    def quantity_units(self) -> List[str]:
        info(f"Accessing '{self.location}' for 'quantity_units'")
        units = []
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantity_names:
                units.append(fp[quant].attrs["UNITS"].astype(str)[0])

        return units

    @cached_property
    def dtype(self) -> np.dtype:
        info(f"Accessing '{self.location}' for 'dtype'")
        fields = []
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantity_names:
                fields.append((quant, fp[quant].dtype))
        return np.dtype(fields)

    @cached_property
    def iteration(self) -> int:
        info(f"Accessing '{self.location}' for 'iteration'")
        with h5.File(self.location, mode="r") as fp:
            iteration = fp.attrs["ITER"][0]
        return iteration

    @cached_property
    def time_step(self) -> float:
        info(f"Accessing '{self.location}' for 'time_step'")
        with h5.File(self.location, mode="r") as fp:
            time_step = fp.attrs["TIME"][0]
        return time_step

    @cached_property
    def time_unit(self) -> str:
        info(f"Accessing '{self.location}' for 'time_unit'")
        with h5.File(self.location, mode="r") as fp:
            time_unit = fp.attrs["TIME UNITS"].astype(str)[0]
        return time_unit


@register_backend(ParticleDataset)
class Osiris_Dev_Hdf5_ParticleFile:
    name = "osiris_dev_particles_hdf5"
    location: Optional[Path] = None

    def __init__(self, location=Union[str, Path]) -> None:
        self.location = (
            location if isinstance(location, Path) else Path(location)
        )

    @staticmethod
    def is_valid_backend(file_path: FileLocation) -> bool:
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

    @cached_property
    def dataset_name(self) -> str:
        info(f"Accessing '{self.location}' for 'dataset_name'")
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["NAME"].astype(str)[0]

    @cached_property
    def num_particles(self) -> int:
        info(f"Accessing '{self.location}' for 'num_particles'")
        with h5.File(self.location, mode="r") as fp:
            return fp["q"].shape[0]

    def get_data(self, indexing=None, fields=None) -> np.ndarray:
        info(f"Reading data in '{self.location}'")
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

    @cached_property
    def quantity_names(self) -> List[str]:
        info(f"Accessing '{self.location}' for 'quantity_names'")
        quantities = []

        with h5.File(self.location, mode="r") as fp:
            for key, item in fp.items():
                if key == "tag":
                    continue
                if isinstance(item, h5.Dataset):
                    quantities.append(key)

        return sort_particle_quantities(quantities, ["x", "p"])

    @cached_property
    def quantity_labels(self) -> List[str]:
        info(f"Accessing '{self.location}' for 'quantity_labels'")
        ordered_quants = self.quantity_names
        labels = []

        with h5.File(self.location, mode="r") as fp:
            unordered_labels = fp.attrs["LABELS"].astype(str)
            unordered_quants = fp.attrs["QUANTS"].astype(str)

            order = []
            for quant in unordered_quants:
                if quant == "tag":
                    continue
                order.append(ordered_quants.index(quant))

            labels = [unordered_labels[i] for i in order]

        return labels

    @cached_property
    def quantity_units(self) -> List[str]:
        info(f"Accessing '{self.location}' for 'quantity_units'")
        ordered_quants = self.quantity_names
        units = []

        with h5.File(self.location, mode="r") as fp:
            unordered_units = fp.attrs["UNITS"].astype(str)
            unordered_quants = fp.attrs["QUANTS"].astype(str)

            order = []
            for quant in unordered_quants:
                if quant == "tag":
                    continue
                order.append(ordered_quants.index(quant))

            units = [unordered_units[i] for i in order]

        return units

    @cached_property
    def dtype(self) -> np.dtype:
        info(f"Accessing '{self.location}' for 'dtype'")
        fields = []
        with h5.File(self.location, mode="r") as fp:
            for quant in self.quantity_names:
                fields.append((quant, fp[quant].dtype))
        return np.dtype(fields)

    @cached_property
    def iteration(self) -> int:
        info(f"Accessing '{self.location}' for 'iteration'")
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["ITER"][0]

    @cached_property
    def time_step(self) -> float:
        info(f"Accessing '{self.location}' for 'time_step'")
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["TIME"][0]

    @cached_property
    def time_unit(self) -> str:
        info(f"Accessing '{self.location}' for 'time_unit'")
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["TIME UNITS"].astype(str)[0]
