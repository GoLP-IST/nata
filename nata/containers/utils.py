# -*- coding: utf-8 -*-
from typing import Iterable
from warnings import warn

import dask.array as da
import numpy as np
from dask import delayed
from numpy.lib import recfunctions

# TODO: remove the use of types
from nata.types import BackendType
from nata.types import DatasetType

from .core import HasNumpyInterface


def is_unique(iterable: Iterable) -> bool:
    return len(set(iterable)) == 1


def register_backend(container: DatasetType):
    """Decorater for registering a backend for a Datset"""
    # if not isinstance(container, DatasetType):
    #     raise TypeError(f"Requires container of type '{DatasetType}'")

    def add_backend_to_container(backend: BackendType):
        if container.is_valid_backend(backend):
            container.add_backend(backend)
            # TODO: make sure it works well with quantities
            #       -> labels are passed appropiate
            backend.__getitem__ = backend.get_data
        else:
            warn(
                f"{backend} is an invalid backend for '{container}'. "
                + "Skipping backend registration!"
            )
        return backend

    return add_backend_to_container


def unstructured_to_structured(data: da.Array, new_dtype: np.dtype) -> da.Array:
    new_shape = data.shape[:-1]
    new_data = delayed(recfunctions.unstructured_to_structured)(data, dtype=new_dtype)
    return da.from_delayed(new_data, new_shape, dtype=new_dtype)


def to_numpy(array_like: HasNumpyInterface) -> np.ndarray:
    if isinstance(array_like, HasNumpyInterface):
        return array_like.to_numpy()
    else:
        raise TypeError(f"requires object of type '{HasNumpyInterface}'")


def to_dask(array_like: HasNumpyInterface) -> da.Array:
    if isinstance(array_like, HasNumpyInterface):
        return array_like.to_dask()
    else:
        raise TypeError(f"requires object of type '{HasNumpyInterface}'")
