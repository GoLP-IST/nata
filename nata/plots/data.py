# -*- coding: utf-8 -*-
import attr
import numpy as np
from attr.validators import instance_of
from attr.validators import optional


@attr.s
class PlotDataAxis:
    name: str = attr.ib(default="")
    label: str = attr.ib(default="")
    units: str = attr.ib(default="")
    data: np.array = attr.ib(
        default=None, validator=optional(instance_of(np.array))
    )
    min: float = attr.ib(default=0, init=False)
    max: float = attr.ib(default=0, init=False)
    type: str = attr.ib(default="", validator=optional(instance_of(str)))

    def __attrs_post_init__(self):
        if self.data is not None:
            self.min = np.min(self.data)
            self.max = np.max(self.data)

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
    units: str = attr.ib(default="", validator=optional(instance_of(str)))
    data: np.ndarray = attr.ib(default=[])
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
