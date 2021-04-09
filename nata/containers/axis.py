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

    @property
    def name(self) -> str:
        return self._name

    @property
    def label(self) -> str:
        return self._label

    @property
    def unit(self) -> str:
        return self._unit
