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

class GridPlot1D(GridPlot):
    def __init__(self, parent=None, axes=None, data=None, show=True, **kwargs):
        
        super().__init__(parent=parent, show=show, **kwargs)

        self.axes=axes
        self.data=data

        self.set_attrs(
            attr_list=[
                ("xlim", (self.axes[0].min, self.axes[0].max)),
                ("ylim", (np.min(self.data.values), np.max(self.data.values))),
                ("xlabel", self.axes[0].get_label()),
                ("ylabel", self.data.get_label()),
                ("title", self.data.get_time_label())
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
        x = self.axes[0].values
        y = self.data.values
        
        # build plot
        self._plt.plot(x, y)

        self._plt.xscale(self._xscale)
        self._plt.yscale(self._yscale)

        self._plt.xlim(self._xlim)
        self._plt.ylim(self._ylim)

        # set axes labels
        self._plt.xlabel(self._xlabel, labelpad=self._pad)
        self._plt.ylabel(self._ylabel, labelpad=self._pad)
        
        # set title
        self._ax.set_title(label=self._title, fontsize=self._fontsize, pad=self._pad)
        
        self.show()

class GridPlot2D(GridPlot):
    def __init__(self, parent=None, axes=None, data=None, show=True, **kwargs):
        
        super().__init__(parent=parent, show=show, **kwargs)

        self.axes=axes
        self.data=data

        self.set_attrs(
            attr_list=[
                ("aspect", "auto"),
                ("xlim", (self.axes[0].min, self.axes[0].max)),
                ("ylim", (self.axes[1].min, self.axes[1].max)),
                ("xlabel", self.axes[0].get_label()),
                ("ylabel", self.axes[1].get_label()),
                ("title", self.data.get_time_label()),
                ("vmin", np.min(self.data.values)),
                ("vmax", np.max(self.data.values)),
                ("cb_map", "rainbow"),
                ("cb_title", self.data.get_label()),
                ("cb_scale", "linear"),
                ("cb_linthresh", 1e-6)
                # ("polar", False),
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
        x = self.axes[0].values
        y = self.axes[1].values
        z = np.transpose(self.data.values)

        # build color map norm
        if   self._cb_scale == "log":
            # convert values to positive numbers
            z = np.abs(z) + 1e-16

            # adjust min and max values
            if self._vmin_auto:
                self._vmin = np.min(z)
            
            if self._vmax_auto:
                self._vmax = np.max(z)

            # set color map norm
            self._cb_norm = clr.LogNorm(
                vmin=self._vmin, 
                vmax=self._vmax
            )
        elif self._cb_scale == "symlog":
            # set color map norm
            self._cb_norm = clr.SymLogNorm(
                vmin=self._vmin, 
                vmax=self._vmax,
                linthresh=self._cb_linthresh
            )
        else:
            self._cb_norm = clr.Normalize(
                vmin=self._vmin, 
                vmax=self._vmax
            )

        # build plot
        self._c = self._ax.pcolormesh(
            x, 
            y, 
            z,
            cmap=self._cb_map,
            norm=self._cb_norm,
            antialiased=False
            )
        
        # draw colorbar
        self._cb = self._plt.colorbar(self._c, aspect=30)
        
        # set colorbar title
        self._cb.set_label(label=self._cb_title, fontsize=self._fontsize, labelpad=self._pad)

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