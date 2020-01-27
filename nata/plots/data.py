import attr
import numpy as np

@attr.s
class PlotData:
    name: str = attr.ib(default="")
    label: str = attr.ib(default="")
    units: str = attr.ib(default="")
    values: np.ndarray = attr.ib(default=[])

    # time properties
    time: float = attr.ib(default=0.)
    time_units: str = attr.ib(default="")

    def get_label(self):
        label = ""
        if self.label:
            label += f"${self.label}$"
            if self.units:
                label += f" $\\left[{self.units}\\right]$"
        return label
    
    def get_time_label(self):
        label = ""
        if self.time is not None:
            label += f"Time = ${self.time:.2f}$"
            if self.time_units:
                label += f" $\\left[{self.time_units}\\right]$"
    
        return label