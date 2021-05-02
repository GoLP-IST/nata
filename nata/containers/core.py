# -*- coding: utf-8 -*-
from functools import partial
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Protocol
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union

import dask.array as da
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class BackendType(Protocol):
    name: str

    @staticmethod
    def is_valid_backend(location: Path) -> bool:
        ...  # pragma: no cover


class HasBackends:
    """Class containing the conceptional logic of backends in nata.

    This class provides an abstraction layer on implementing the backend logic in nata.
    To use this logic, any class can use this class as a parent class to incoporate
    backends.
    """

    _backend_protocol: BackendType
    _backends: Set[BackendType]

    def __init_subclass__(
        cls,
        *,
        backend_protocol: Optional[BackendType] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init_subclass__(**kwargs)

        if backend_protocol is None:
            raise AttributeError("requires backend protocol")

        cls._backend_protocol = backend_protocol
        cls._backends = set()

    @classmethod
    def add_backend(cls, backend: BackendType) -> None:
        """Add backend to class.

        Class method which takes a backend and add it to the set of available backends
        for the class.

        Arguments:
            backend: Backend which will be added to the class.

        Raises:
            TypeError: Raised when provided backend is invalid
        """
        if cls.is_valid_backend(backend):
            cls._backends.add(backend)
        else:
            raise TypeError("invalid backend provided")

    @classmethod
    def remove_backend(cls, backend: Union[BackendType, str]) -> None:
        """Remove backend from a class.

        Class method which removes a backend from a set of class-specific backends.

        Arguments:
            backend: Backend or name of the backend which will be removed.

        Raises:
            ValueError: Raised when provided backend or name do not match registered
                        backends.
        """
        if isinstance(backend, str):
            for b in cls._backends:
                if b.name == backend:
                    cls._backends.remove(b)
                    break
            else:
                raise ValueError(f"backend '{backend}' not registered")
        else:
            try:
                cls._backends.remove(backend)
            except KeyError:
                raise ValueError(f"backend '{backend}' not registered")

    @classmethod
    def is_valid_backend(cls, backend: BackendType) -> bool:
        """Check if a backend is a valid backend.

        Any class which uses a backend systems can only accept a specific backend. This
        backend is determined by the backend protocol `cls._backend_protocol` of a
        class.

        Arguments:
            backend: Backend which will be checked if it fulfills the backend protocol
                     of the class.
        """
        return isinstance(backend, cls._backend_protocol)

    @classmethod
    def get_backends(cls) -> Dict[str, BackendType]:
        """Return a dictionary of registred backends."""
        backends_dict = {}
        for backend in cls._backends:
            backends_dict[backend.name] = backend
        return backends_dict

    @classmethod
    def get_valid_backend(cls, path: Path) -> Optional[BackendType]:
        """Returns a valid backend for a given file location."""
        for backend in cls._backends:
            if backend.is_valid_backend(path):
                return backend
        else:
            return None


class HasNumpyInterface(NDArrayOperatorsMixin):
    _handled_array_ufunc: Dict[np.ufunc, Callable]
    _handled_array_function: Dict[Callable, Callable]

    _data: da.Array

    def __init_subclass__(cls, **kwargs: Dict[str, Any]) -> None:
        super().__init_subclass__(**kwargs)

        cls._handled_array_ufunc = {}
        cls._handled_array_function = {}

        if not hasattr(cls, "from_array"):
            raise NotImplementedError("'from_array' method is not implemented")

    @classmethod
    def get_handled_array_ufunc(cls) -> Dict[np.ufunc, Callable]:
        return cls._handled_array_ufunc

    @classmethod
    def add_handled_array_ufunc(
        cls, function: np.ufunc, implementation: Callable
    ) -> None:
        if not isinstance(function, np.ufunc):
            raise TypeError("provided function is not of type ufunc")

        cls._handled_array_ufunc[function] = implementation

    @classmethod
    def remove_handled_array_ufunc(cls, function: np.ufunc) -> None:
        if function not in cls._handled_array_ufunc:
            raise ValueError(f"ufunc '{function}' is not registered")

        del cls._handled_array_ufunc[function]

    @classmethod
    def get_handled_array_function(cls) -> Dict[Callable, Callable]:
        return cls._handled_array_function

    @classmethod
    def add_handled_array_function(
        cls, function: Callable, implementation: Callable
    ) -> None:
        cls._handled_array_function[function] = implementation

    @classmethod
    def remove_handeld_array_function(cls, function: Callable) -> None:
        if function not in cls._handled_array_function:
            raise ValueError(f"function '{function}' is not registered")

        del cls._handled_array_function[function]

    @classmethod
    def implements(cls, function: Union[np.ufunc, Callable]):
        def decorator(func):
            if isinstance(function, np.ufunc):
                cls._handled_array_ufunc[function] = func
            else:
                cls._handled_array_function[function] = func

            return func

        return decorator

    def to_dask(self) -> da.Array:
        return self._data

    def to_numpy(self) -> np.ndarray:
        return self._data.compute()

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    def __len__(self) -> int:
        return len(self._data)

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        arr = self.to_numpy()

        # '__array__' requires to return ndarray and 'to_numpy' can produce other types
        if not isinstance(arr, np.ndarray):
            arr = np.asanyarray(arr)

        return arr.astype(dtype) if dtype else arr

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    ) -> "HasNumpyInterface":
        if ufunc in self._handled_array_ufunc:
            return self._handled_array_ufunc[ufunc](method, *inputs, **kwargs)

        # repack inputs to mimic passing as dask array
        repacked_inputs = ()
        for input_ in inputs:
            if input_ is self:
                repacked_inputs += (self._data,)
            elif isinstance(input_, HasNumpyInterface):
                repacked_inputs += (input_.to_dask(),)
            else:
                repacked_inputs += (input_,)

        # required additional repacking if in-place
        if "out" in kwargs:
            output = tuple(self._data if arg is self else arg for arg in kwargs["out"])
            kwargs["out"] = output

        data = self._data.__array_ufunc__(ufunc, method, *repacked_inputs, **kwargs)

        # Reraise 'NotImplemented' if does dask does not implement it and let numpy take
        # over in the dispatching process:
        #
        # https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
        if data is NotImplemented:
            raise NotImplemented  # noqa: F901

        # Handle in-place operations scenario
        elif data is None:
            self._data = kwargs["out"][0]
            return self
        else:
            return self.from_array(data)

    def __array_function__(
        self,
        func: Callable,
        types: Tuple[Type],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> "HasNumpyInterface":
        if func in self._handled_array_function:
            return self._handled_array_function[func](types, args, kwargs)

        # repack arguments
        types = tuple(type(self._data) if t is type(self) else t for t in types)
        args = tuple(self._data if arg is self else arg for arg in args)
        data = self._data.__array_function__(func, types, args, kwargs)

        return self.from_array(data)


class HasPluginSystem:
    _plugin_as_property: Dict[str, Callable]
    _plugin_as_method: Dict[str, Callable]

    def __init_subclass__(cls, **kwargs: Dict[str, Any]) -> None:
        super().__init_subclass__(**kwargs)

        cls._plugin_as_property = {}
        cls._plugin_as_method = {}

        cls._something = None

    def __getattribute__(self, name: str) -> Any:
        if name == "_plugin_as_property":
            return super().__getattribute__(name)

        if name == "_plugin_as_method":
            return super().__getattribute__(name)

        if name in self._plugin_as_property:
            return self._plugin_as_property[name](self)

        if name in self._plugin_as_method:
            func = partial(self._plugin_as_method[name], self)
            func.__doc__ = self._plugin_as_method[name].__doc__
            return func

        return super().__getattribute__(name)

    @classmethod
    def get_property_plugin(cls) -> Dict[str, Callable]:
        return cls._plugin_as_property

    @classmethod
    def add_property_plugin(cls, plugin_name: str, plugin: Callable) -> None:
        if not isinstance(plugin_name, str):
            raise TypeError("'plugin_name' has to be a 'str'")

        if not plugin_name.isidentifier():
            raise ValueError(f"'{plugin_name}' has to be a valid identifier")

        cls._plugin_as_property[plugin_name] = plugin

    @classmethod
    def remove_property_plugin(cls, plugin_name: str) -> None:
        if plugin_name not in cls._plugin_as_property:
            raise ValueError(f"plugin '{plugin_name}' is not registered")

        del cls._plugin_as_property[plugin_name]

    @classmethod
    def get_method_plugin(cls) -> Dict[str, Callable]:
        return cls._plugin_as_method

    @classmethod
    def add_method_plugin(cls, plugin_name: str, plugin: Callable) -> None:
        if not isinstance(plugin_name, str):
            raise TypeError("'plugin_name' has to be a 'str'")

        if not plugin_name.isidentifier():
            raise ValueError(f"'{plugin_name}' has to be a valid identifier")

        cls._plugin_as_method[plugin_name] = plugin

    @classmethod
    def remove_method_plugin(cls, plugin_name: str) -> None:
        if plugin_name not in cls._plugin_as_method:
            raise ValueError(f"plugin '{plugin_name}' is not registered")

        del cls._plugin_as_method[plugin_name]
