# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np


class Axis(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(
        self,
        data: da.Array,
        *,
        name: str = "unnamed",
        label: str = "unlabeled",
        unit: str = "",
        has_appendable_dim: bool = False,
    ) -> None:
        self._data = data if isinstance(data, da.Array) else da.asanyarray(data)
        self._has_appendable_dim = has_appendable_dim

        if not name.isidentifier():
            raise ValueError("Argument 'name' has to be a valid identifier")

        self._name = name
        self._label = label
        self._unit = unit

    def __repr__(self) -> str:
        return f"Axis(name='{self.name}', label='{self.label}', unit='{self.unit}')"

    def _repr_html_(self) -> str:
        return (
            "<span>Axis</span>"
            "<span style='color: var(--jp-info-color0);'>"
            "("
            f"name='{self.name}', "
            f"label='{self.label}', "
            f"unit='{self.unit}'"
            ")"
            "</span>"
        )

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        if not new_name.isidentifier():
            raise ValueError("New name has to be a valid identifier")

        self._name = new_name

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, new_label: str) -> None:
        self._label = new_label

    @property
    def unit(self) -> str:
        return self._unit

    @unit.setter
    def unit(self, new_unit: str) -> None:
        self._unit = new_unit

    def as_dask(self) -> da.Array:
        return self._data
