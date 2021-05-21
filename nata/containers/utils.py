# -*- coding: utf-8 -*-
import inspect
from functools import partial
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union
from warnings import warn

import dask.array as da
import numpy as np
from dask import delayed
from numpy.lib import recfunctions

# TODO: remove the use of types
from nata.types import BackendType
from nata.types import DatasetType

from .core import HasNumpyInterface
from .core import HasPluginSystem


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


def _annotation_of_first_arg(func: Callable) -> Union[Any, HasPluginSystem]:
    func_signature = inspect.signature(func)
    return next(iter(func_signature.parameters.values())).annotation


def register_plugin(
    func_or_else: Optional[
        Union[Callable, HasPluginSystem, Sequence[HasPluginSystem]]
    ] = None,
    *,
    name: Optional[str] = None,
    plugin_type: Literal["method", "property"] = "method",
):
    # Helper functions
    def add_func_to_containers(seq: Sequence[HasPluginSystem], func: Callable):
        for obj in seq:
            if plugin_type == "method":
                obj.add_method_plugin(name or func.__name__, func)
            else:
                obj.add_property_plugin(name or func.__name__, func)

        return func

    def container_in_annotation(func: Callable):
        obj = _annotation_of_first_arg(func)
        # special case Union[A, B]
        if hasattr(obj, "__args__") and all(
            issubclass(a, HasPluginSystem) for a in obj.__args__
        ):
            add_func_to_containers(obj.__args__, func)
        elif issubclass(obj, HasPluginSystem):
            add_func_to_containers((obj,), func)
        else:
            raise TypeError(f"requires a subclass of {HasPluginSystem}")

        return func

    # Plugin registration
    if func_or_else:
        if inspect.isfunction(func_or_else):
            return container_in_annotation(func_or_else)

        elif isinstance(func_or_else, Sequence) and all(
            issubclass(c, HasPluginSystem) for c in func_or_else
        ):
            return partial(add_func_to_containers, func_or_else)

        elif issubclass(func_or_else, HasPluginSystem):
            return partial(add_func_to_containers, (func_or_else,))

        else:
            raise TypeError("invalid type for registration provided")

    else:
        return container_in_annotation
