# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import h5py as h5
import numpy as np

from nata.containers import GridDataset
from nata.containers import ParticleDataset
from nata.utils.container_tools import register_backend


@register_backend(GridDataset)
class Osiris_Hdf5_GridFile:
    name = "osiris_4.4.4_grid_hdf5"
    location: Optional[Path] = None

    def __init__(self, location: Union[str, Path]) -> None:
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
            if (
                ("NAME" in f.attrs)
                and ("TYPE" in f.attrs)
                and ("LABEL" not in f.attrs)
            ):
                type_: str = f.attrs["TYPE"].astype(str)[0]
                # general naming
                name_: str = f.attrs["NAME"].astype(str)[0]
                names: Tuple[str, ...] = (name_,)
                # special case naming
                name_ = name_.split()[-1]
                name_ = name_.replace("_", "")

                names += (name_,)
                if (type_ == "grid") and any([name in f for name in names]):
                    return True

        return False

    @property
    def _dset_name(self) -> str:
        with h5.File(self.location, mode="r") as fp:
            short_name = fp.attrs["NAME"].astype(str)[0]
            if short_name in fp:
                return short_name

            name_ = short_name.split()[-1]
            name_ = name_.replace("_", "")
            if name_ in fp:
                return name_

    def get_data(self, indexing=None):
        # TODO: apply indexing here
        with h5.File(self.location, mode="r") as fp:
            dset = fp[self._dset_name]
            dataset = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(dataset)
        return dataset.transpose()

    @property
    def dataset_name(self) -> str:
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["NAME"].astype(str)[0]

    @property
    def dataset_label(self) -> str:
        with h5.File(self.location, mode="r") as fp:
            return fp[self._dset_name].attrs["LONG_NAME"].astype(str)[0]

    @property
    def ndim(self):
        with h5.File(self.location, mode="r") as fp:
            ndim = fp[self._dset_name].ndim
        return ndim

    @property
    def shape(self):
        with h5.File(self.location, mode="r") as fp:
            return fp[self._dset_name].shape[::-1]

    @property
    def dtype(self):
        with h5.File(self.location, mode="r") as fp:
            dtype = fp[self._dset_name].dtype
        return dtype

    @property
    def dataset_unit(self):
        with h5.File(self.location, mode="r") as fp:
            units = fp[self._dset_name].attrs["UNITS"].astype(str)[0]
        return units

    @property
    def axes_min(self):
        min_ = []
        with h5.File(self.location, mode="r") as fp:
            for axis in fp["AXIS"]:
                min_.append(fp["AXIS/" + axis][0])
        return np.array(min_)

    @property
    def axes_max(self):
        max_ = []
        with h5.File(self.location, mode="r") as fp:
            for axis in fp["AXIS"]:
                max_.append(fp["AXIS/" + axis][1])

        return np.array(max_)

    @property
    def axes_names(self):
        names = []
        with h5.File(self.location, mode="r") as fp:
            for axis in fp["AXIS"]:
                names.append(fp["AXIS/" + axis].attrs["NAME"].astype(str)[0])
        return np.array(names)

    @property
    def axes_labels(self):
        long_names = []
        with h5.File(self.location, mode="r") as fp:
            for axis in fp["AXIS"]:
                long_names.append(
                    fp["AXIS/" + axis].attrs["LONG_NAME"].astype(str)[0]
                )
        return np.array(long_names)

    @property
    def axes_units(self):
        units = []
        with h5.File(self.location, mode="r") as fp:
            for axis in fp["AXIS"]:
                units.append(fp["AXIS/" + axis].attrs["UNITS"].astype(str)[0])
        return np.array(units)

    @property
    def iteration(self):
        with h5.File(self.location, mode="r") as fp:
            time_step = fp.attrs["ITER"].astype(int)[0]
        return time_step

    @property
    def time_step(self):
        with h5.File(self.location, mode="r") as fp:
            time = fp.attrs["TIME"][0]
        return time

    @property
    def time_unit(self):
        with h5.File(self.location, mode="r") as fp:
            time_unit = fp.attrs["TIME UNITS"].astype(str)[0]
        return time_unit


@register_backend(GridDataset)
class Osiris_Dev_Hdf5_GridFile:
    name = "osiris_dev_grid_hdf5"
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

        if not file_path.suffix == ".h5":
            return False

        if not h5.is_hdf5(file_path):
            return False

        with h5.File(file_path, mode="r") as f:
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

    @property
    def _dset_name(self) -> str:
        with h5.File(self.location, mode="r") as fp:
            short_name = fp.attrs["NAME"].astype(str)[0]
            if short_name in fp:
                return short_name

            name_ = short_name.split()[-1]
            name_ = name_.replace("_", "")
            if name_ in fp:
                return name_

    def get_data(self, indexing=None):
        # TODO: apply indexing here
        with h5.File(self.location, mode="r") as fp:
            dset = fp[self._dset_name]
            dataset = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(dataset)
        return dataset.transpose()

    @property
    def dataset_name(self) -> str:
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["NAME"].astype(str)[0]

    @property
    def dataset_label(self) -> str:
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["LABEL"].astype(str)[0]

    @property
    def ndim(self):
        with h5.File(self.location, mode="r") as fp:
            ndim = fp[self._dset_name].ndim
        return ndim

    @property
    def shape(self):
        with h5.File(self.location, mode="r") as fp:
            return fp[self._dset_name].shape[::-1]

    @property
    def dtype(self):
        with h5.File(self.location, mode="r") as fp:
            dtype = fp[self._dset_name].dtype
        return dtype

    @property
    def dataset_unit(self):
        with h5.File(self.location, mode="r") as fp:
            units = fp.attrs["UNITS"].astype(str)[0]
        return units

    @property
    def axes_min(self):
        min_ = []
        with h5.File(self.location, mode="r") as fp:
            for axis in fp["AXIS"]:
                min_.append(fp["AXIS/" + axis][0])
        return np.array(min_)

    @property
    def axes_max(self):
        max_ = []
        with h5.File(self.location, mode="r") as fp:
            for axis in fp["AXIS"]:
                max_.append(fp["AXIS/" + axis][1])

        return np.array(max_)

    @property
    def axes_names(self):
        names = []
        with h5.File(self.location, mode="r") as fp:
            for axis in fp["AXIS"]:
                names.append(fp["AXIS/" + axis].attrs["NAME"].astype(str)[0])
        return np.array(names)

    @property
    def axes_labels(self):
        long_names = []
        with h5.File(self.location, mode="r") as fp:
            for axis in fp["AXIS"]:
                long_names.append(
                    fp["AXIS/" + axis].attrs["LONG_NAME"].astype(str)[0]
                )
        return np.array(long_names)

    @property
    def axes_units(self):
        units = []
        with h5.File(self.location, mode="r") as fp:
            for axis in fp["AXIS"]:
                units.append(fp["AXIS/" + axis].attrs["UNITS"].astype(str)[0])
        return np.array(units)

    @property
    def iteration(self):
        with h5.File(self.location, mode="r") as fp:
            time_step = fp.attrs["ITER"].astype(int)[0]
        return time_step

    @property
    def time_step(self):
        with h5.File(self.location, mode="r") as fp:
            time = fp.attrs["TIME"][0]
        return time

    @property
    def time_unit(self):
        with h5.File(self.location, mode="r") as fp:
            time_unit = fp.attrs["TIME UNITS"].astype(str)[0]
        return time_unit


@register_backend(ParticleDataset)
class Osiris_Hdf5_ParticleFile:
    name = "osiris_4.4.4_particles_hdf5"
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


@register_backend(ParticleDataset)
class Osiris_Dev_Hdf5_ParticleFile:
    name = "osiris_dev_particles_hdf5"
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
