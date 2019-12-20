from pathlib import Path
from typing import Dict, Set, Union

import attr
import numpy as np

from nata.containers import GridDataset, ParticleDataset, location_exist
from nata.containers.types import DatasetTypes
from nata.utils.exceptions import NataInvalidContainer
from nata.utils.info_printer import PrettyPrinter


@attr.s
class DatasetCollection:
    root_path: Path = attr.ib(converter=Path, validator=location_exist)
    _container_list: Set[DatasetTypes] = set([GridDataset, ParticleDataset])
    store: Dict[str, DatasetTypes] = attr.ib(factory=dict)

    def info(self, full: bool = False):
        printer = PrettyPrinter(header="Collection")
        printer.add_line(f"Root path: {self.root_path}")
        printer.add_line(f"Number of datasets: {len(self.store)}")
        printer.add_line(f"Datasets: " + ", ".join(self.datasets))
        printer.new_linebreak()

        if full:
            for dset in self.store.values():
                dset.info(printer=printer, root_path=self.root_path)

        printer.flush()

    @property
    def datasets(self):
        return np.array(['"' + k + '"' for k in self.store.keys()], dtype=str)

    # TODO: check if it's possible to append direct a DataSet
    def append(self, obj: Union[str, Path, "DatasetCollection"]) -> None:
        """Takes a path to a diagnostic and appends it to the collection."""
        if not isinstance(obj, (str, Path, DatasetCollection)):
            raise ValueError(
                f"Can not append object of type '{type(obj)}' to collection"
            )

        # If it is a DatasetCollection itself
        if isinstance(obj, DatasetCollection):
            self.store.update(obj.store)
            return

        # If it is a path or string -> initialize Dataset Object
        for container in self._container_list:
            try:
                dataset = container(obj)
                break
            except NataInvalidContainer:
                continue
        else:
            return  # TODO: check if warning might be better

        if dataset.name in self.store:
            existing_ds = self.store[dataset.name]

            if existing_ds.appendable:
                existing_ds.append(dataset)
            else:
                raise ValueError(
                    f"Dataset '{existing_ds.name}' is not appendable!"
                )

        else:
            self.store[dataset.name] = dataset

    def __getitem__(self, key):
        return self.store[key]

    @classmethod
    def register_plugin(cls, plugin_name, plugin):
        setattr(cls, plugin_name, plugin)