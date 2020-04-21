# -*- coding: utf-8 -*-
from copy import copy
from typing import List
from typing import Optional

import matplotlib as mpl

# from nata.plots import Figure
from nata.plots import PlotData
from nata.plots import PlotTypes


class Axes:
    def __init__(
        self,
        fig,
        index: int = 0,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        xscale: Optional[str] = "linear",
        yscale: Optional[str] = "linear",
        aspect: Optional[str] = "auto",
        legend_show: Optional[bool] = True,
        legend_loc: Optional[str] = "upper right",
        legend_frameon: Optional[bool] = False,
        cb_show: Optional[bool] = True,
        cb_loc: Optional[str] = "right",
    ):

        self._fig = fig
        self.index = index

        self.xlim = xlim
        self.xlim_auto = None

        self.ylim = xlim
        self.ylim_auto = None

        self.xlabel = xlabel
        self.xlabel_auto = None

        self.ylabel = ylabel
        self.ylabel_auto = None

        self.title = title
        self.title_auto = None

        self.xscale = xscale
        self.yscale = xscale

        self.aspect = aspect

        self.legend_show = legend_show
        self.legend_loc = legend_loc
        self.legend_frameon = legend_frameon

        self.cb_show = cb_show
        self.cb_loc = cb_loc

        # initialize list of plot objects
        self.init_plots()

        # initialize axes backend
        self.init_backend()

    @property
    def plots(self) -> list:
        return self._plots

    # backend methods
    # TODO: generalize the following methods for arbitrary backend
    def init_backend(self):
        with mpl.rc_context(fname=self._fig.fname, rc=self._fig.rc):
            self._ax = self._fig._fig.add_subplot(
                self._fig.nrows, self._fig.ncols, self.index
            )

        self.legend = None
        self.cb = None

    def clear_backend(self):
        for plot in self._plots:
            plot.clear()

        self.clear_colorbar()
        self.clear_legend()

        self._ax.clear()
        self._ax.remove()
        self._ax = None

    def reset_backend(self):
        self.clear_backend()
        self.init_backend()

    def update_backend(self):
        with mpl.rc_context(fname=self._fig.fname, rc=self._fig.rc):
            ax = self._ax

            ax.set_xscale(self.xscale)
            ax.set_yscale(self.yscale)

            if self.xlim[0] != self.xlim[1]:
                ax.set_xlim(self.xlim)

            if self.ylim[0] != self.ylim[1]:
                ax.set_ylim(self.ylim)

            # set axes labels
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)

            # set title
            ax.set_title(label=self.title)

            # set aspect ratio
            ax.set_aspect(self.aspect)

    # plot methods
    def init_plots(self):
        self._plots = []
        self._cb = None
        self._legend = None

    def add_plot(
        self,
        plot_type: Optional[PlotTypes] = None,
        plot: Optional[PlotTypes] = None,
        data: Optional[PlotData] = None,
        style: Optional[dict] = dict(),
    ):
        if plot is None:
            plot = plot_type(axes=self, data=data, **style)
        else:
            plot._axes = self

        self._plots.append(plot)

        return plot

    def update_plot_options(self):
        if self.xlim_auto:
            self.xlim = (
                min([p._default_xlim()[0] for p in self._plots]),
                max([p._default_xlim()[1] for p in self._plots]),
            )

        if self.ylim_auto:
            self.ylim = (
                min([p._default_ylim()[0] for p in self._plots]),
                max([p._default_ylim()[1] for p in self._plots]),
            )

        if self.xlabel_auto:
            units = [p._xunits() for p in self._plots]
            xlabels = [p._default_xlabel(units=False) for p in self._plots]

            if len(set(units)) == 1 and len(set(xlabels)) == 1:
                self.xlabel = set(xlabels).pop() + " [" + set(units).pop() + "]"
            elif len(set(units)) == 1:
                xlabels = [p._default_xlabel(units=False) for p in self._plots]
                self.xlabel = ", ".join(xlabels) + " [" + set(units).pop() + "]"
            else:
                xlabels = [p._default_xlabel(units=True) for p in self._plots]
                self.xlabel = ", ".join(xlabels)

        if self.ylabel_auto:
            units = [p._yunits() for p in self._plots]
            ylabels = [p._default_ylabel(units=False) for p in self._plots]

            if len(set(units)) == 1 and len(set(ylabels)) == 1:
                self.ylabel = set(ylabels).pop() + " [" + set(units).pop() + "]"
            elif len(set(units)) == 1:
                ylabels = [p._default_ylabel(units=False) for p in self._plots]
                self.ylabel = ", ".join(ylabels) + " [" + set(units).pop() + "]"
            else:
                ylabels = [p._default_ylabel(units=True) for p in self._plots]
                self.ylabel = ", ".join(ylabels)

        if self.title_auto:
            titles = [p._default_title() for p in self._plots]

            if len(set(titles)) == 1:
                self.title = set(titles).pop()
            else:
                self.title = None

    def update(self):
        self.update_plot_options()
        self.update_backend()

        if len(self._plots) > 1:
            self.init_legend()

    def redo_plots(self):

        self.reset_backend()

        for plot in self.plots:
            plot.build_canvas()

        self.update()

    # legend methods
    def init_legend(self):
        if self.legend_show:
            handles, labels = self._ax.get_legend_handles_labels()
            with mpl.rc_context(fname=self._fig.fname, rc=self._fig.rc):
                # show legend
                self._legend = self._ax.legend(
                    handles=handles,
                    labels=labels,
                    loc=self.legend_loc,
                    frameon=self.legend_frameon,
                )

    def clear_legend(self):
        if self._legend:
            self._legend.remove()

    # colorbar methods
    def init_colorbar(self, plot: PlotTypes):
        if self.cb_show:
            with mpl.rc_context(fname=self._fig.fname, rc=self._fig.rc):
                # show colorbar
                self._cb = self._ax.get_figure().colorbar(
                    plot._h, location=self.cb_loc, aspect=30
                )

                # set colorbar title
                self._cb.set_label(label=plot.cb_title)

    def clear_colorbar(self):
        if self._cb:
            self._cb.remove()

    def copy(self):
        new = copy(self)

        for plot in new._plots:
            plot._axes = new

        return new

    @classmethod
    def style_attrs(self) -> List[str]:
        return [
            "xlim",
            "ylim",
            "xlabel",
            "ylabel",
            "title",
            "xscale",
            "yscale",
            "aspect",
            "legend_show",
            "legend_loc",
            "legend_frameon",
            "cb_show",
            "cb_loc",
        ]
