import attr
import numpy as np
import matplotlib.colors as clr

from nata.plots.base import BasePlot
# from nata.plots.figure import Figure

@attr.s
class GridPlot1D(BasePlot):

    def _default_xlim(self):
        return (self.axes[0].min, self.axes[0].max)
    
    def _default_ylim(self):
        return (np.min(self.data.values), np.max(self.data.values))
    
    def _default_xlabel(self):
        return self.axes[0].get_label()
    
    def _default_ylabel(self):
        return self.data.get_label()

    def _default_title(self):
        return self.data.get_time_label()

    def __attrs_post_init__(self):
        
        self.build_canvas()

    def build_canvas(self):

        # add subplot
        ax = self._backend_fig.add_subplot(self.fig_pos)

        # get plot axes and data
        x = self.axes[0].values
        y = self.data.values
        
        # build plot
        ax.plot(x, y)

        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        # set axes labels
        ax.set_xlabel(self.xlabel, labelpad=self.fig.pad)
        ax.set_ylabel(self.ylabel, labelpad=self.fig.pad)
        
        # set title
        ax.set_title(label=self.title, pad=self.fig.pad)

        # set aspect ratio
        ax.set_aspect(self.aspect)
        
        # set backend axes object
        self._ax = ax
        
@attr.s
class GridPlot2D(BasePlot):
    
    # TODO: validate vmin and vmax with any real number type
    vmin: float = attr.ib(
        # validator=attr.validators.instance_of((int, float))
    )
    vmax: float = attr.ib(
        # validator=attr.validators.instance_of((int, float, np.float))
    )
    cb_map: str = attr.ib(
        default="rainbow", 
        validator=attr.validators.instance_of(str)
    )
    cb_title: str = attr.ib(
        validator=attr.validators.instance_of(str)
    )
    cb_scale: str = attr.ib(
        default="linear", 
        validator=attr.validators.instance_of(str)
    )
    cb_linthresh: float = attr.ib(
        default=1e-6, 
        validator=attr.validators.instance_of((int, float))
    )

    def _default_xlim(self):
        return (self.axes[0].min, self.axes[0].max)
    
    def _default_ylim(self):
        return (self.axes[1].min, self.axes[1].max)
    
    def _default_xlabel(self):
        return self.axes[0].get_label()
    
    def _default_ylabel(self):
        return self.axes[1].get_label()

    def _default_title(self):
        return self.data.get_time_label()

    @vmin.default
    def _default_vmin(self):
        return np.min(self.data.values)

    @vmax.default
    def _default_vmax(self):
        return np.max(self.data.values)

    @cb_title.default
    def _default_cb_title(self):
        return self.data.get_label()


    def __attrs_post_init__(self):
        
        self.build_canvas()

    def build_canvas(self):

        # add subplot
        ax = self._backend_fig.add_subplot(self.fig_pos)

        # get plot axes and data
        x = self.axes[0].values
        y = self.axes[1].values
        z = np.transpose(self.data.values)

        # build color map norm
        if   self.cb_scale == "log":
            # convert values to positive numbers
            z = np.abs(z) + 1e-16

            # adjust min and max values
            # TODO: do this only if vmin was not init
            # self.vmin = np.min(z)
            
            # if self.vmax_auto:
            # self.vmax = np.max(z)

            # set color map norm
            self.cb_norm = clr.LogNorm(
                vmin=self.vmin, 
                vmax=self.vmax
            )
        elif self.cb_scale == "symlog":
            # set color map norm
            self.cb_norm = clr.SymLogNorm(
                vmin=self.vmin, 
                vmax=self.vmax,
                linthresh=self.cb_linthresh
            )
        else:
            self.cb_norm = clr.Normalize(
                vmin=self.vmin, 
                vmax=self.vmax
            )

        # build plot
        c = ax.pcolormesh(
            x, 
            y, 
            z,
            cmap=self.cb_map,
            norm=self.cb_norm,
            antialiased=False
            )
        
        # draw colorbar
        cb = ax.get_figure().colorbar(c, aspect=30)
        
        # set colorbar title
        cb.set_label(label=self.cb_title, labelpad=self.fig.pad)

        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        # set axes labels
        ax.set_xlabel(self.xlabel, labelpad=self.fig.pad)
        ax.set_ylabel(self.ylabel, labelpad=self.fig.pad)
        
        # set title
        ax.set_title(label=self.title, pad=self.fig.pad)
        
        # set aspect ratio
        ax.set_aspect(self.aspect)

        # set backend axes object
        self._ax = ax