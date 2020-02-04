# -*- coding: utf-8 -*-
import attr
import numpy as np


@attr.s
class PlotDataAxis:
    name: str = attr.ib(default="")
    label: str = attr.ib(default="")
    units: str = attr.ib(default="")
    min: float = attr.ib(default=0)
    max: float = attr.ib(default=0)
    n: int = attr.ib(default=0)
    type: str = attr.ib(default="")

    def __attrs_post_init__(self):
        # build plot data axis values
        self.set_values()

    def set_values(self):
        if self.type == "linear":
            self.values = np.linspace(start=self.min, stop=self.max, num=self.n)

        elif self.type == "log":
            base = (self.max / self.min) ** (1 / self.n)
            self.values = self.min * np.logspace(
                start=0, stop=self.n, num=self.n, base=base
            )

    def get_label(self, units=True):
        label = ""
        if self.label:
            label += f"${self.label}$"
            if units and self.units:
                label += f" $\\left[{self.units}\\right]$"
        return label


@attr.s
class PlotData:
    name: str = attr.ib(default="")
    label: str = attr.ib(default="")
    units: str = attr.ib(default="")
    values: np.ndarray = attr.ib(default=[])
    axes: list = attr.ib(default=())

    # time properties
    time: float = attr.ib(default=0.0)
    time_units: str = attr.ib(default="")

    def get_label(self, units=True):
        label = ""
        if self.label:
            label += f"${self.label}$"
            if units and self.units:
                label += f" $\\left[{self.units}\\right]$"
        return label

    def get_time_label(self):
        label = ""
        if self.time is not None:
            label += f"Time = ${self.time:.2f}$"
            if self.time_units:
                label += f" $\\left[{self.time_units}\\right]$"

        return label
