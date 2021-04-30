# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Protocol
from typing import Set
from typing import Union

from dask.array import Array
from numpy import ufunc
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
    _handled_array_ufunc: Dict[ufunc, Callable]
    _handled_array_function: Dict[Callable, Callable]

    _data: Array

    def __init_subclass__(cls, **kwargs: Dict[str, Any]) -> None:
        super().__init_subclass__(**kwargs)

        cls._handled_array_ufunc = {}
        cls._handled_array_function = {}

    @classmethod
    def get_handled_array_ufunc(cls) -> Dict[ufunc, Callable]:
        return cls._handled_array_ufunc

    @classmethod
    def add_handled_array_ufunc(cls, function: ufunc, implementation: Callable) -> None:
        if not isinstance(function, ufunc):
            raise TypeError("provided function is not of type ufunc")

        cls._handled_array_ufunc[function] = implementation

    @classmethod
    def remove_handeld_array_ufunc(cls, function: ufunc) -> None:
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
