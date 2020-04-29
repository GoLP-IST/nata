# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List
from typing import Optional

import numpy as np


@dataclass
class PlotDataAxis:
    name: str = ""
    label: Optional[str] = None
    units: Optional[str] = None
    data: Optional[np.ndarray] = None

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


@dataclass
class PlotData:
    data: np.ndarray
    axes: List[PlotDataAxis]
    name: str = ""
    label: Optional[str] = None
    units: Optional[str] = None
    time: Optional[float] = None
    time_units: Optional[str] = None

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
