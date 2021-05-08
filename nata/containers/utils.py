# -*- coding: utf-8 -*-
from warnings import warn

import dask.array as da
import numpy as np
from dask import delayed
from numpy.lib import recfunctions

# TODO: remove the use of types
from nata.types import BackendType
from nata.types import DatasetType


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
