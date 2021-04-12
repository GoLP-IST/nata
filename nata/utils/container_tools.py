# -*- coding: utf-8 -*-
from warnings import warn

from nata.types import BackendType
from nata.types import DatasetType


def register_backend(container: DatasetType):
    """Decorater for registering a backend for a Datset"""
    # if not isinstance(container, DatasetType):
    #     raise TypeError(f"Requires container of type '{DatasetType}'")

    def add_backend_to_container(backend: BackendType):
        if container.is_valid_backend(backend):
            container.add_backend(backend)
            backend.__getitem__ = backend.get_data
        else:
            warn(
                f"{backend} is an invalid backend for '{container}'. "
                + "Skipping backend registration!"
            )
        return backend

    return add_backend_to_container
