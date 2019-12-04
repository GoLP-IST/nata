import numpy as np

import matplotlib.colors as clr

from .base import BasePlot

class GridPlot(BasePlot):
    def __init__(self, parent=None, show=True, **kwargs):
        
        super().__init__(parent=parent, show=show, **kwargs)

        self.set_attrs(
            attr_list=[
                ("xscale", "linear"),
                ("yscale", "linear"),
            ],
            kwargs=kwargs
        )

class ParticlePlot2D(GridPlot):
    def __init__(self, parent=None, sel=None, axes=None, show=True, **kwargs):
        
        super().__init__(parent=parent, show=show, **kwargs)

        self.sel=sel
        self.axes=axes

        self.set_attrs(
            attr_list=[
                ("aspect", "auto"),
                ("xlim", (self.axes[0].min, self.axes[0].max)),
                ("ylim", (self.axes[1].min, self.axes[1].max)),
                ("xlabel", self.axes[0].get_label()),
                ("ylabel", self.axes[1].get_label()),
                ("title", ""),
            ],
            kwargs=kwargs
        )

        self.build_canvas()

    def build_canvas(self):

        # create figure
        self._plt.ioff()
        self._fig = self._plt.figure(figsize=self._figsize, facecolor="#ffffff")
        self._ax = self._fig.add_subplot(111)

        # get plot axes and data
        x = self._parent.data[self.sel[0]]
        y = self._parent.data[self.sel[1]]

        # build plot
        self._plt.plot(x, y, ",")

        self._plt.xscale(self._xscale)
        self._plt.yscale(self._yscale)

        self._plt.xlim(self._xlim)
        self._plt.ylim(self._ylim)

        # set axes labels
        self._plt.xlabel(self._xlabel, labelpad=self._pad)
        self._plt.ylabel(self._ylabel, labelpad=self._pad)
        
        # set title
        self._ax.set_title(label=self._title, fontsize=self._fontsize, pad=self._pad)
        
        # set aspect ratio
        self._ax.set_aspect(self._aspect)

        self.show()