# -*- coding: utf-8 -*-
from abc import ABC
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Set

import attr
from attr import validators

from nata.containers import location_exist


def register_backend(container):
    if not issubclass(container, BaseDataset):
        raise ValueError("Invalid container passed for backend registration!")

    def add_to_backend(backend):
        container.add_backend(backend)
        return backend

    return add_to_backend


@attr.s(init=False)
class BaseDataset(ABC):
    _backends: Set[Any] = set()
    appendable = False

    location: Optional[Path] = attr.ib(
        validator=validators.optional(location_exist), eq=False, order=False
    )

    @classmethod
    def register_plugin(cls, plugin_name, plugin):
        setattr(cls, plugin_name, plugin)

    @classmethod
    def add_backend(cls, backend):
        cls._backends.add(backend)
