import numpy as np

class PlotAxis:

    def __init__(self, name="", label="", units="", xtype="", xmin=0, xmax=0, nx=0):
        self.name  = name
        self.label = label

        self.units = units
        self.min   = xmin
        self.max   = xmax
        self.n     = nx
        self.type  = xtype
        
        self.set_values()

    def set_values(self):
        if   self.type == "linear":
            self.values = np.linspace(start=self.min, stop=self.max, num=self.n)

        elif self.type == 'log':
            base = (self.max/self.min) ** (1/self.n)
            self.values = self.min * np.logspace(start=0, stop=self.n, num=self.n, base=base)

    def get_label(self):
        label = ""
        if self.label:
            label += f"${self.label}$"
            if self.units:
                label += f" $\\left[{self.units}\\right]"
            label += "$"
        return label