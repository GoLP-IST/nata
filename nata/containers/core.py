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

    def __init__(self, data: da.Array) -> None:
        if not isinstance(data, da.Array):
            raise TypeError(f"'data' has to be of type '{type(da.Array)}'")

        self._data = data

    def __init_subclass__(cls, **kwargs: Dict[str, Any]) -> None:
        super().__init_subclass__(**kwargs)

        cls._handled_array_ufunc = {}
        cls._handled_array_function = {}

    @classmethod
    def from_array(cls, data: da.Array) -> "HasNumpyInterface":
        return cls(data)

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
    def get_property_plugins(cls) -> Dict[str, Callable]:
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
    def get_method_plugins(cls) -> Dict[str, Callable]:
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


class HasAnnotations:
    _name: str
    _label: str
    _unit: str

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new: str) -> None:
        new = new if isinstance(new, str) else str(new, encoding="utf-8")
        if not new.isidentifier():
            raise ValueError("'name' has to be an identifier")
        self._name = new

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, new: str) -> None:
        self._label = new

    @property
    def unit(self) -> str:
        return self._unit

    @unit.setter
    def unit(self, new: str) -> None:
        self._unit = new


class HasName:
    _name: str
    _label: str

    def __init__(self, name: str, label: str) -> None:
        if not isinstance(name, str):
            raise TypeError("'name' has to be of type 'str'")

        if not isinstance(label, str):
            raise TypeError("'label' has to be of type 'str'")

        if not name.isidentifier():
            raise ValueError("'name' has to be valid identifier")

        self._name = name
        self._label = label

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        if not isinstance(new_name, str):
            raise TypeError("'new_name' has to be of type 'str'")

        if not new_name.isidentifier():
            raise ValueError("'name' has to be valid identifier")

        self._name = new_name

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, new_label: str) -> None:
        if not isinstance(new_label, str):
            raise TypeError("'new_label' has to be of type 'str'")

        self._label = new_label


class HasUnit:
    _unit: str

    def __init__(self, unit: str) -> None:
        if not isinstance(unit, str):
            raise TypeError("'unit' has to be of type 'str'")

        self._unit = unit

    @property
    def unit(self) -> str:
        return self._unit

    @unit.setter
    def unit(self, new: str) -> None:
        self._unit = new


# TODO: make quantitites indexable and allow to change name, label, and units
class HasQuantities:
    _quantities: Tuple[Tuple[str, str, str], ...]

    def __init__(self, quantities: Tuple[Tuple[str, str, str], ...]) -> None:
        if (
            not isinstance(quantities, tuple)
            or not all(isinstance(q, tuple) for q in quantities)
            or not all(isinstance(v, str) for q in quantities for v in q)
            or not all(len(q) == 3 for q in quantities)
        ):
            raise TypeError(
                "'quantities' has to be a 'Tuple[Tuple[str, str, str], ...]'"
            )

        self._quantities = quantities

    @property
    def quantities(self) -> Tuple[Tuple[str, str, str], ...]:
        return self._quantities

    @property
    def quantity_names(self) -> Tuple[str, ...]:
        return tuple(q[0] for q in self._quantities)

    @property
    def quantity_labels(self) -> Tuple[str, ...]:
        return tuple(q[1] for q in self._quantities)

    @property
    def quantity_units(self) -> Tuple[str, ...]:
        return tuple(q[2] for q in self._quantities)


class HasCount:
    _count: int

    def __init__(self, count: int) -> None:
        if not isinstance(count, int):
            raise TypeError("'count' has to be 'int'")

        self._count = count

    @property
    def count(self) -> int:
        return self._count
