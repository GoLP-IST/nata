from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional
from math import ceil

import attr
import numpy as np


# TODO: add type annotations and consider renaming properties
@attr.s
class BaseGrid(ABC):
    location: Path = attr.ib(converter=Path)
    _selection: Optional[Tuple[slice]] = attr.ib(init=False, default=None)

    @staticmethod
    def shape_from_slice(old_shape, slice_of_array):
        # makes function call consistent! -> If None occurs return simply
        # previous shape
        if slice_of_array is None:
            return old_shape

        # sanity conversion
        s = np.index_exp[slice_of_array]

        # iterate over each dimension
        new_shape = tuple()
        for len_, slice_ in zip(old_shape, s):
            if isinstance(slice_, int):
                continue
            # unpack slice -> if negative consider length
            if slice_.start:
                start = (
                    slice_.start if slice_.start >= 0 else (len_ + slice_.start)
                )
            else:
                start = 0

            if slice_.stop:
                stop = slice_.stop if slice_.stop >= 0 else (len_ + slice_.stop)
            else:
                stop = len_

            if slice_.step:
                step = slice_.step if slice_.step else 1
            else:
                step = 1

            if stop <= start and step > 0:
                new_shape += (0,)
            else:
                new_shape += (ceil((stop - start) / step),)

        return new_shape

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, new_selection):
        if len(new_selection) != len(self.shape):
            raise ValueError(
                f"Selection of {self.dim}d data with {len(new_selection)} slices is ambiguous!"
            )

        if self._selection:
            # TODO: add selection of selection
            raise NotImplementedError("Not yet implemented")
        else:
            self._selection = new_selection

    @staticmethod
    @abstractmethod
    def is_valid_backend(file_path):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def short_name(self):
        pass

    @property
    @abstractmethod
    def long_name(self):
        pass

    @property
    @abstractmethod
    def dataset(self):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    @abstractmethod
    def dtype(self):
        pass

    @property
    @abstractmethod
    def dataset_unit(self):
        pass

    @property
    @abstractmethod
    def axis_min(self):
        pass

    @property
    @abstractmethod
    def axis_max(self):
        pass

    @property
    @abstractmethod
    def axes_names(self):
        pass

    @property
    @abstractmethod
    def axes_long_names(self):
        pass

    @property
    @abstractmethod
    def axes_units(self):
        pass

    @property
    @abstractmethod
    def iteration(self):
        pass

    @property
    @abstractmethod
    def time_step(self):
        pass

    @property
    @abstractmethod
    def time_unit(self):
        pass
