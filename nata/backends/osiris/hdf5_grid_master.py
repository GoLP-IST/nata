from pathlib import Path
from typing import Union

import h5py as h5
import numpy as np

from nata.backends import BaseGrid
from nata.containers import GridDataset, register_backend


@register_backend(GridDataset)
class Osiris_Hdf5_GridFile_Master(BaseGrid):
    name = "osiris_hdf5_grid"

    @staticmethod
    def is_valid_backend(file_path: Path) -> bool:
        if not file_path.is_file():
            return False

        if not file_path.suffix == ".h5":
            return False

        if not h5.is_hdf5(file_path):
            return False

        with h5.File(file_path, mode="r") as f:
            if ("NAME" in f.attrs) and \
               ("TYPE" in f.attrs) and \
               ("LABEL" in f.attrs):
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

    @property
    def short_name(self) -> str:
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["NAME"].astype(str)[0]

    @property
    def long_name(self) -> str:
        with h5.File(self.location, mode="r") as fp:
            return fp.attrs["LABEL"].astype(str)[0]

    @property
    def dataset(self):
        with h5.File(self.location, mode="r") as fp:
            dset = fp[self._dset_name]
            shape = self.shape_from_slice(dset.shape, self._selection)
            dataset = np.zeros(shape, dtype=dset.dtype)
            dset.read_direct(dataset, source_sel=self._selection, dest_sel=None)
        return dataset.transpose()

    @property
    def dim(self):
        with h5.File(self.location, mode="r") as fp:
            ndim = fp[self._dset_name].ndim
        return ndim

    @property
    def shape(self):
        with h5.File(self.location, mode="r") as fp:
            shape = fp[self._dset_name].shape[::-1]
        return self.shape_from_slice(shape, self._selection)

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
    def axis_min(self):
        # if selection is given -> do interpolation by creating a dummy
        # linspace array
        # TODO: remove array creation and turn in simple calculation
        if self.selection:
            min_values = []
            max_values = []
            with h5.File(self.location, mode="r") as fp:
                for axis in fp["AXIS"]:
                    min_values.append(fp["AXIS/" + axis][0])
                    max_values.append(fp["AXIS/" + axis][1])
                dset = fp[self.short_name]
                shape = dset.shape

            min_ = []
            for i, N in enumerate(shape):
                if isinstance(self._selection[i], int):
                    min_.append(
                        np.linspace(min_values[i], max_values[i], N)[
                            self._selection
                        ]
                    )
                    continue
                min_.append(
                    np.linspace(min_values[i], max_values[i], N)[
                        self._selection
                    ][0]
                )

        else:
            min_ = []
            with h5.File(self.location, mode="r") as fp:
                for axis in fp["AXIS"]:
                    min_.append(fp["AXIS/" + axis][0])

        return np.array(min_)

    @property
    def axis_max(self):
        if self.selection:
            min_values = []
            max_values = []
            with h5.File(self.location, mode="r") as fp:
                for axis in fp["AXIS"]:
                    min_values.append(fp["AXIS/" + axis][0])
                    max_values.append(fp["AXIS/" + axis][1])
                dset = fp[self.short_name]
                shape = dset.shape

            max_ = []
            for i, N in enumerate(shape):
                if isinstance(self._selection[i], int):
                    max_.append(
                        np.linspace(min_values[i], max_values[i], N)[
                            self._selection
                        ]
                    )
                    continue
                max_.append(
                    np.linspace(min_values[i], max_values[i], N)[
                        self._selection
                    ][-1]
                )

        else:
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
    def axes_long_names(self):
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
            time_step = fp.attrs["ITER"][0]
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
