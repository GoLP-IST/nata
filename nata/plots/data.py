import numpy as np

class PlotData:

    def __init__(self, name="", label="", units="", data=None, time=0., time_units=""):
        self.name   = name
        self.label  = label
        self.units  = units
        self.values = data

        self.time = time
        self.time_units = time_units

    def get_label(self):
        label = ""
        if self.label:
            label += f"${self.label}$"
            if self.units:
                label += f" $\\left[{self.units}\\right]"
            label += "$"
        return label
    
    def get_time_label(self):
         # TODO: move this to the dataset object

        label = ""
        if self.time is not None:
            label += f"Time = ${self.time:.2f}"
            if self.time_units:
                label += f" [{self.time_units}]"
            label += "$"
    
        return label  