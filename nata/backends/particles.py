# -*- coding: utf-8 -*-
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import attr
import numpy as np
from attr import converters
from attr import validators
from numpy.lib.recfunctions import rename_fields
from numpy.lib.recfunctions import unstructured_to_structured

from nata.utils.attrs import is_identifier
from nata.utils.attrs import subdtype_of


@attr.s
class ParticleBackend(ABC):
    location: Optional[Path] = attr.ib(
        default=None, converter=converters.optional(Path)
    )

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
    def dataset_name(self):
        pass

    @property
    @abstractmethod
    def num_particles(self):
        pass

    @property
    @abstractmethod
    def has_tags(self):
        pass

    @property
    @abstractmethod
    def tags(self):
        pass

    @abstractmethod
    def get_data(self, indexing, fields):
        pass

    @property
    @abstractmethod
    def quantities(self):
        pass

    @property
    @abstractmethod
    def quantity_labels(self):
        pass

    @property
    @abstractmethod
    def quantity_units(self):
        pass

    @property
    @abstractmethod
    def dtype(self):
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


_PART_BACKEND_ARRAY_COUNT = 0


def _convert_to_structured_array(arg):
    arr = np.asanyarray(arg)

    if not arr.dtype.names:
        if arr.ndim == 1:
            arr = arr.reshape((1, len(arr)))
        arr = np.swapaxes(arr, 0, -1)
        arr = unstructured_to_structured(arr)
        # rename default fields
        arr = rename_fields(
            arr, {f: f"quant{i+1}" for i, f in enumerate(arr.dtype.names)}
        )
    return arr


@attr.s
class ParticleArray(ParticleBackend):
    name = "ParticleArray"
    location = attr.ib(default=None, init=False)
    _array: np.ndarray = attr.ib(
        default=None,
        converter=_convert_to_structured_array,
        validator=validators.instance_of(np.ndarray),
        eq=False,
        order=False,
    )

    _dataset_name: str = attr.ib(
        default=None,
        validator=validators.optional(
            [validators.instance_of(str), is_identifier]
        ),
    )
    _num_particles: int = attr.ib(default=None)

    _quantities: np.ndarray = attr.ib(
        default=None,
        converter=converters.optional(np.asanyarray),
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=subdtype_of(np.str_),
                iterable_validator=validators.instance_of(np.ndarray),
            )
        ),
    )
    _quantity_labels: np.ndarray = attr.ib(
        default=None,
        converter=converters.optional(np.asanyarray),
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=subdtype_of(np.str_),
                iterable_validator=validators.instance_of(np.ndarray),
            )
        ),
    )
    _quantity_units: np.ndarray = attr.ib(
        default=None,
        converter=converters.optional(np.asanyarray),
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=subdtype_of(np.str_),
                iterable_validator=validators.instance_of(np.ndarray),
            )
        ),
    )
    _iteration: np.int = attr.ib(
        default=None,
        eq=False,
        order=False,
        validator=validators.optional(subdtype_of(np.integer)),
    )
    _time_step: np.int = attr.ib(
        default=None,
        eq=False,
        order=False,
        validator=validators.optional(subdtype_of(np.floating)),
    )
    _time_unit: str = attr.ib(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    _keep_creation_count: bool = attr.ib(default=False)

    @_array.validator
    def _only_1d_and_2d(self, attrib, value):
        if value.ndim != 1:
            raise ValueError("Generated array should be a 1d array with fields")

    @_quantities.validator
    @_quantity_labels.validator
    @_quantity_units.validator
    def _matches_with_array_dim(self, attrib, value):
        # mimics optional validator
        if value is None:
            return

        if len(value) != len(self._array.dtype.names):
            raise ValueError(
                f"Mismatch in number of elements for `{attr.name.strip('_')}"
            )

    def __attrs_post_init__(self):
        global _PART_BACKEND_ARRAY_COUNT

        if self._dataset_name is None:
            if not self._keep_creation_count:
                _PART_BACKEND_ARRAY_COUNT += 1
            self._dataset_name = f"prt_array_{_PART_BACKEND_ARRAY_COUNT}"

        if self._quantities is None:
            self._quantities = np.array(self._array.dtype.names)
        else:
            rename_fields(
                self._array,
                {
                    old: new
                    for old, new in zip(
                        self._array.dtype.names, self._quantities
                    )
                },
            )

        if self._quantity_labels is None:
            self._quantity_labels = self._quantities.copy()

        if self._quantity_units is None:
            self._quantity_units = np.array(
                ["" for _ in range(len(self._quantities))]
            )

        if self._iteration is None:
            self._iteration = 0

        if self._time_step is None:
            self._time_step = 0.0

        if self._time_unit is None:
            self._time_unit = ""

        del self._keep_creation_count

        # other default params which has to be there - to be consistent
        self._tags = None
        self._has_tags = False

    @staticmethod
    def is_valid_backend(file_path):
        False

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def num_particles(self):
        return len(self._dataset)

    @property
    def has_tags(self):
        return self._has_tags

    @property
    def tags(self):
        return self._tags

    def get_data(self, indexing, fields):
        return self._array[fields][indexing]

    @property
    def quantities(self):
        return self._quantities

    @property
    def quantity_labels(self):
        return self._quantity_labels

    @property
    def quantity_units(self):
        return self._quantity_units

    @property
    def dtype(self):
        return self._dataset.dtype

    @property
    def iteration(self):
        return self._iteration

    @property
    def time_step(self):
        return self._time_step

    @property
    def time_unit(self):
        return self._time_unit
