# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union

import numpy as np

from nata.utils.exceptions import NataInvalidContainer

from .grid import GridDataset
from .ParticleDataset import ParticleDataset


class DatasetCollection:
    _container_set = set([GridDataset, ParticleDataset])

    def __init__(self, root_path: Union[str, Path]) -> None:
        self.root_path = root_path if isinstance(root_path, Path) else Path(root_path)
        self.store = dict()

    def __repr__(self) -> str:
        try:
            path = self.root_path.relative_to(Path().absolute())
        except ValueError:
            path = self.root_path

        repr_ = f"{self.__class__.__name__}("
        repr_ += f"root_path='{path}', "
        repr_ += f"stored={[k for k in self.store]}"
        repr_ += ")"

        return repr_

    def info(self, full: bool = False):  # pragma: no cover
        return self.__repr__()

    @property
    def datasets(self):
        return np.array([k for k in self.store.keys()], dtype=str)

    def _append_datasetcollection(self, obj):
        self.store.update(obj.store)

    def _append_file(self, obj):
        for container in self._container_set:
            try:
                dataset = container(obj)
                break
            except NataInvalidContainer:
                continue
        else:
            # not possible to append the file -> not a valid container found
            return

        if dataset.name in self.store:
            existing_ds = self.store[dataset.name]
            existing_ds.append(dataset)
        else:
            self.store[dataset.name] = dataset

    def append(self, obj: Union[str, Path, "DatasetCollection"]) -> None:
        """Takes a path to a diagnostic and appends it to the collection."""
        if isinstance(obj, DatasetCollection):
            self._append_datasetcollection(obj)

        elif isinstance(obj, (str, Path)):
            self._append_file(obj)

        elif isinstance(obj, (GridDataset, ParticleDataset)):
            self.store[obj.name] = obj
        else:
            raise ValueError(
                f"Can not append object of type '{type(obj)}' to collection"
            )

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    @classmethod
    def register_plugin(cls, plugin_name, plugin):
        setattr(cls, plugin_name, plugin)
