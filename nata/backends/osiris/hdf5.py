# -*- coding: utf-8 -*-
from logging import info
from pathlib import Path
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import h5py as h5
import numpy as np

import ndindex as ndx
from nata.containers import GridDataset
from nata.containers import ParticleDataset
from nata.types import BasicIndex
from nata.types import BasicIndexing
from nata.types import FileLocation
from nata.utils.backends import sort_particle_quantities
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

    def get_data(self, indexing: Optional[BasicIndexing] = None) -> np.ndarray:
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

    def get_data(self, indexing: Optional[BasicIndexing] = None) -> np.ndarray:
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

    def __init__(self, location: FileLocation) -> None:
        self.location = (
            location if isinstance(location, Path) else Path(location)
        )

        info(f"Obtaining backend props for '{self.location}'")
        with h5.File(self.location, mode="r") as fp:
            self._dataset_name = fp.attrs["NAME"].astype(str)[0]
            self._num_particles = fp["q"].shape[0] if fp["q"].shape else 0

            # find first all quantaties - with their names
            names = []
            for key, item in fp.items():
                if key == "tag":
                    continue

                if isinstance(item, h5.Dataset):
                    names.append(key)

            self._quantity_names = sort_particle_quantities(names, ("x", "p"))

            # iterate over all the sorted names and fill other props
            labels = []
            units = []
            dtype = []
            for name in self._quantity_names:
                labels.append(fp[name].attrs["LONG_NAME"].astype(str)[0])
                units.append(fp[name].attrs["UNITS"].astype(str)[0])
                dtype.append((name, fp[name].dtype))

            self._quantity_labels = labels
            self._quantity_units = units
            self._dtype = np.dtype(dtype)

            # temporal information
            self._iteration = fp.attrs["ITER"][0]
            self._time_step = fp.attrs["TIME"][0]
            self._time_unit = fp.attrs["TIME UNITS"].astype(str)[0]

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def num_particles(self) -> int:
        return self._num_particles

    @property
    def quantity_names(self) -> List[str]:
        return self._quantity_names

    @property
    def quantity_labels(self) -> List[str]:
        return self._quantity_labels

    @property
    def quantity_units(self) -> List[str]:
        return self._quantity_units

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def time_step(self) -> float:
        return self._time_step

    @property
    def time_unit(self) -> str:
        return self._time_unit

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
            if ("TYPE" in f.attrs) and ("LABELS" not in f.attrs):
                if f.attrs["TYPE"].astype(str)[0] == "particles":
                    return True

        return False

    def get_data(
        self,
        indexing: Optional[BasicIndex] = None,
        fields: Optional[Union[str, Sequence[str]]] = None,
    ) -> np.ndarray:
        info(f"Reading data in '{self.location}'")
        with h5.File(self.location, mode="r") as fp:
            index = (
                ndx.Slice(None) if indexing is None else ndx.ndindex(indexing)
            )
            index = index.reduce((self.num_particles,))
            dtype = self.dtype if fields is None else self.dtype[fields]

            if dtype.fields:
                # create array as source to store quantities
                dset = np.empty(len(index), dtype=dtype)

                for field in dtype.fields:
                    dset[field][:] = fp[field][index.raw]
            else:
                # only one field element -> string passed
                dset = fp[fields][index.raw]

        return dset


@register_backend(ParticleDataset)
class Osiris_Dev_Hdf5_ParticleFile:
    name = "osiris_dev_particles_hdf5"
    location: Optional[Path] = None

    def __init__(self, location=Union[str, Path]) -> None:
        self.location = (
            location if isinstance(location, Path) else Path(location)
        )

        info(f"Obtaining backend props for '{self.location}'")
        with h5.File(self.location, mode="r") as fp:
            self._dataset_name = fp.attrs["NAME"].astype(str)[0]
            self._num_particles = fp["q"].shape[0] if fp["q"].shape else 0

            # quantaties
            unordered_names = list(fp.attrs["QUANTS"].astype(str))
            unordered_labels = list(fp.attrs["LABELS"].astype(str))
            unordered_units = list(fp.attrs["UNITS"].astype(str))

            clean_names = [name for name in unordered_names if name != "tag"]
            clean_names = sort_particle_quantities(clean_names, ("x", "p"))

            clean_labels = [
                unordered_labels[unordered_names.index(s)] for s in clean_names
            ]
            clean_units = [
                unordered_units[unordered_names.index(s)] for s in clean_names
            ]
            dtype = [(name, fp[name].dtype) for name in clean_names]

            self._quantity_names = clean_names
            self._quantity_labels = clean_labels
            self._quantity_units = clean_units
            self._dtype = np.dtype(dtype)

            # temporal information
            self._iteration = fp.attrs["ITER"][0]
            self._time_step = fp.attrs["TIME"][0]
            self._time_unit = fp.attrs["TIME UNITS"].astype(str)[0]

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def num_particles(self) -> int:
        return self._num_particles

    @property
    def quantity_names(self) -> List[str]:
        return self._quantity_names

    @property
    def quantity_labels(self) -> List[str]:
        return self._quantity_labels

    @property
    def quantity_units(self) -> List[str]:
        return self._quantity_units

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def time_step(self) -> float:
        return self._time_step

    @property
    def time_unit(self) -> str:
        return self._time_unit

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
            if ("TYPE" in f.attrs) and ("LABELS" in f.attrs):
                if f.attrs["TYPE"].astype(str)[0] == "particles":
                    return True

        return False

    def get_data(
        self,
        indexing: Optional[BasicIndex] = None,
        fields: Optional[Union[str, Sequence[str]]] = None,
    ) -> np.ndarray:
        info(f"Reading data in '{self.location}'")
        with h5.File(self.location, mode="r") as fp:
            index = (
                ndx.Slice(None) if indexing is None else ndx.ndindex(indexing)
            )
            index = index.reduce((self.num_particles,))
            dtype = self.dtype if fields is None else self.dtype[fields]

            if dtype.fields:
                # create array as source to store quantities
                dset = np.empty(len(index), dtype=dtype)

                for field in dtype.fields:
                    dset[field][:] = fp[field][index.raw]
            else:
                # only one field element -> string passed
                dset = fp[fields][index.raw]

        return dset
