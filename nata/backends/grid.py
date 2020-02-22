# -*- coding: utf-8 -*-
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import attr
import numpy as np
from attr import converters
from attr import validators


@attr.s
class BaseGrid(ABC):
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
    def dataset_label(self):
        pass

    @abstractmethod
    def get_data(self, indexing):
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
    def axes_min(self):
        pass

    @property
    @abstractmethod
    def axes_max(self):
        pass

    @property
    @abstractmethod
    def axes_names(self):
        pass

    @property
    @abstractmethod
    def axes_labels(self):
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


_ARRAY_CREATION_COUNT = 0


def _matches_with_dim(instance, attribute, value):
    if len(value.shape) != 1:
        raise ValueError(f"ambiguous dimension for {attribute.name}")

    if value.shape[0] != len(instance.array.shape):
        raise ValueError(f"insufficient specification for {attribute.name}")


def _is_identifier(instance, attribute, value):
    if_raise = False
    if isinstance(value, np.ndarray):
        for v in value:
            if not v.isidentifier():
                if_raise = True
    else:
        if not value.isidentifier():
            if_raise = True

    if if_raise:
        raise ValueError(
            f"attribute {attribute.name.strip('_')} "
            + f"has an invalid string '{value}'"
        )


@attr.s
class GridArray(BaseGrid):
    name = "GridArray"
    location = attr.ib(default=None, init=False)
    array: np.ndarray = attr.ib(
        default=None,
        validator=validators.instance_of(np.ndarray),
        eq=False,
        order=False,
    )
    _dataset_name: str = attr.ib(
        default=None,
        validator=validators.optional(
            [validators.instance_of(str), _is_identifier]
        ),
    )
    _dataset_label: str = attr.ib(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    _dataset_unit: str = attr.ib(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    _axes_names: np.ndarray = attr.ib(
        default=None,
        converter=converters.optional(np.array),
        validator=validators.optional([_matches_with_dim, _is_identifier]),
    )
    _axes_labels: np.ndarray = attr.ib(
        default=None,
        converter=converters.optional(np.array),
        validator=validators.optional(_matches_with_dim),
    )
    _axes_min: np.ndarray = attr.ib(
        default=None,
        converter=converters.optional(np.array),
        validator=validators.optional(_matches_with_dim),
        eq=False,
        order=False,
    )
    _axes_max: np.ndarray = attr.ib(
        default=None,
        converter=converters.optional(np.array),
        validator=validators.optional(_matches_with_dim),
        eq=False,
        order=False,
    )
    _axes_units: np.ndarray = attr.ib(
        default=None,
        converter=converters.optional(np.array),
        validator=validators.optional(_matches_with_dim),
    )
    _iteration: int = attr.ib(default=0, converter=int, eq=False, order=False)
    _time_step: float = attr.ib(
        default=0.0, converter=float, eq=False, order=False
    )
    _time_unit: str = attr.ib(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    _keep_creation_count: bool = attr.ib(default=False)

    def __attrs_post_init__(self):
        global _ARRAY_CREATION_COUNT

        if self._dataset_name is None:
            if not self._keep_creation_count:
                _ARRAY_CREATION_COUNT += 1
            self._dataset_name = f"array_{_ARRAY_CREATION_COUNT}"

        # drop first dimension
        # e.g. 1d array can be (1, x) but should be (x,)
        if self.array.shape[0] == 1 and len(self.array.shape) > 1:
            self.array = self.array.reshape(self.array.shape[1:])

        if self._dataset_label is None:
            self._dataset_label = ""

        if self._dataset_unit is None:
            self._dataset_unit = ""

        if self._axes_names is None:
            self._axes_names = np.array([f"x{i+1}" for i in range(self.dim)])

        if self._axes_labels is None:
            self._axes_labels = np.array([f"x_{i+1}" for i in range(self.dim)])

        if self._axes_min is None:
            self._axes_min = np.array([0.0 for _ in range(self.dim)])

        if self._axes_max is None:
            self._axes_max = np.array([1.0 for i in range(self.dim)])

        if self._axes_units is None:
            self._axes_units = np.array(["" for i in range(self.dim)])

        if self._time_unit is None:
            self._time_unit = ""

    @staticmethod
    def is_valid_backend(obj):
        if isinstance(obj, np.ndarray):
            return True
        return False

    @property
    def get_data(self, indexing):
        return self._array[indexing]

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def dataset_label(self):
        return self._dataset_label

    @property
    def dim(self):
        return self.array.ndim

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def dataset_unit(self):
        return self._dataset_unit

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
