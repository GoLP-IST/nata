from abc import ABC, abstractmethod
from pathlib import Path
from typing import Set, Any, Optional

import attr

from nata.containers import location_exist
from nata.utils.info_printer import PrettyPrinter


def register_backend(container):
    if not issubclass(container, BaseDataset):
        raise ValueError("Invalid container passed for backend registration!")

    def add_to_backend(backend):
        container.add_backend(backend)
        return backend

    return add_to_backend


@attr.s
class BaseDataset(ABC):
    _backends: Set[Any] = set()
    appendable = False

    location: Path = attr.ib(
        converter=Path, validator=location_exist, repr=False
    )

    @classmethod
    def register_plugin(cls, plugin_name, plugin):
        setattr(cls, plugin_name, plugin)

    @classmethod
    def add_backend(cls, backend):
        cls._backends.add(backend)

    @abstractmethod
    def info(
        self,
        printer: Optional[PrettyPrinter] = None,
        root_path: Optional[Path] = None,
    ):  # pragma: no cover
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:  # pragma: no cover
        pass
