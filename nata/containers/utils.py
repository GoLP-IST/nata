# -*- coding: utf-8 -*-
from typing import Callable
from warnings import warn

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
            backend.__getitem__ = backend.get_data
        else:
            warn(
                f"{backend} is an invalid backend for '{container}'. "
                + "Skipping backend registration!"
            )
        return backend

    return add_backend_to_container


def get_doc_heading(func: Callable) -> str:
    docs = func.__doc__

    if not docs:
        return ""

    for line in docs.split("\n"):
        if line:
            return line.strip()

    return ""
