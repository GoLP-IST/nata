# -*- coding: utf-8 -*-
from typing import Any
from typing import List
from typing import Set

import numpy as np

from nata.utils.exceptions import NataInvalidContainer


def register_backend(container):
    if not issubclass(container, BaseDataset):
        raise ValueError("Invalid container passed for backend registration!")

    def add_to_backend(backend):
        container.add_backend(backend)
        return backend

    return add_to_backend


# TODO: specify data type that it is a list of np.ndarrays and backends with
#       read_data method
def convert_unstructured_data_to_array(
    data: List[Any], dtype, indices, fields=None
):
    # read out arrays if they are not in the list
    for i, elem in enumerate(data):
        if not isinstance(elem, np.ndarray):
            if fields is None:
                data[i] = elem.get_data(indices)
            else:
                data[i] = elem.get_data(indices, fields)

    # `copy=False` should never be fulfilled and a copy should be done
    #              we use it here just in case if we can avoid a copy, the rest
    #              is done by numpy as it should check internally
    return np.squeeze(np.array(data, dtype=dtype, copy=False))


class BaseDataset:
    _backends: Set[Any] = set()
    appendable = False

    @classmethod
    def register_plugin(cls, plugin_name, plugin):
        setattr(cls, plugin_name, plugin)

    @classmethod
    def add_backend(cls, backend):
        cls._backends.add(backend)

    def _convert_to_backend(self, obj):
        for backend in self._backends:
            if backend.is_valid_backend(obj):
                return backend(obj)

        raise NataInvalidContainer(
            f"Unable to find proper backend for {type(obj)}"
        )

    def _check_appendability(self, other: "BaseDataset"):
        if not self.appendable:
            raise TypeError(f"'{self.__class__}' is not appendable")

        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Can not append '{type(other)}' to '{self.__class__}'"
            )

        if self != other:
            raise ValueError(
                f"Can not append different '{self.__class__.__name__}'"
            )
