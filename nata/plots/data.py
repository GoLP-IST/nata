# -*- coding: utf-8 -*-
from typing import List
from typing import Optional
from typing import Union

import numpy as np


class PlotDataAxis:
    def __init__(
        self,
        name: str = "",
        label: Optional[str] = None,
        units: Optional[str] = None,
        data: Optional[np.ndarray] = None,
    ):

        self.name = name
        self.label = label
        self.units = units
        self.data = None if data is None else np.asanyarray(data)

    @property
    def min(self) -> float:
        return np.min(self.data)

    @property
    def max(self) -> float:
        return np.max(self.data)

    def get_label(self, units=True) -> str:
        label = ""
        if self.label:
            label += f"${self.label}$"
            if units and self.units:
                label += f" $\\left[{self.units}\\right]$"
        return label


class PlotData:
    def __init__(
        self,
        data: np.ndarray,
        axes: List[PlotDataAxis],
        name: str = "",
        label: Optional[str] = None,
        units: Optional[str] = None,
        time: Optional[Union[float, int]] = None,
        time_units: Optional[str] = None,
    ):
        self.data = np.asanyarray(data)
        self.axes = axes
        self.name = name
        self.label = label
        self.units = units
        self.time = time
        self.time_units = time_units

    def get_label(self, units=True) -> str:
        label = ""
        if self.label:
            label += f"${self.label}$"
            if units and self.units:
                label += f" $\\left[{self.units}\\right]$"
        return label

    def get_time_label(self) -> str:
        label = ""
        if self.time is not None:
            label += f"Time = ${self.time:.2f}$"
            if self.time_units:
                label += f" $\\left[{self.time_units}\\right]$"

        return label
