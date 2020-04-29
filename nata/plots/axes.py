# -*- coding: utf-8 -*-
from copy import copy
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import List
from typing import Optional

import matplotlib as mpl

# from nata.plots import Figure
from nata.plots.data import PlotData
from nata.plots.types import PlotTypes


@dataclass
class Axes:
    """Container of parameters and parent and child objects (including\
    plotting backend-related objects) relevant to draw a figure axes.

    Parameters
    ----------
    xlim: ``tuple``, optional
        Limits of the horizontal axis in the format ``(min,max)``. If not
        provided, it is inferred from the dataset(s) represented in the axes.

    ylim: ``tuple``, optional
        Same as ``xlim`` for the vertical axis.

    xlabel: ``str``, optional
        Label of the horizontal axis. If not provided, it is inferred from the
        dataset(s) represented in the axes.

    ylabel: ``str``, optional
        Same as ``xlabel`` for the vertical axis.

    xscale: ``{'linear','log', 'symlog'}``, optional
        Scale of the horizontal axes. If not provided, defaults to
        ``'linear'``.

    yscale: ``{'linear','log', 'symlog'}``, optional
        Same as ``xscale`` for the vertical axis.

    title: ``str``, optional
        Axes title. If not provided, it is inferred from the dataset(s)
        represented in the axes.

    legend_show: ``bool``, optional
        Controls the visibility of the axes legend, when applicable. If not
        provided, defaults to ``True``.

    legend_loc: ``str``, optional
        Controls the position of the axes legend, when applicable. See
        :meth:`matplotlib.axes.Axes.legend` for available options. If not
        provided, defaults to ``'upper right'``.

    legend_frameon: ``bool``, optional
        Controls the visibility of the axes legend frame. If not provided,
        defaults to ``False``.

    cb_show: ``bool``, optional
        Controls the visibility of the axes colorbar, when applicable. If not
        provided, defaults to ``True``.

    index: ``int``, optional
        Position of the axes in the parent figure. Must be between ``0`` and
        ``N-1``, where ``N`` is the number of child axes objects in the parent
        figure. Increases along rows before columns.

    """

    # style properties
    xlim: Optional[tuple] = None
    ylim: Optional[tuple] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    title: Optional[str] = None
    xscale: Optional[str] = "linear"
    yscale: Optional[str] = "linear"
    aspect: Optional[str] = "auto"
    legend_show: Optional[bool] = True
    legend_loc: Optional[str] = "upper right"
    legend_frameon: Optional[bool] = False
    cb_show: Optional[bool] = True

    # other parameters
    index: Optional[int] = 0

    # flags for automatic style properties
    xlim_auto: bool = field(init=False, default=None)
    ylim_auto: bool = field(init=False, default=None)
    xlabel_auto: bool = field(init=False, default=None)
    ylabel_auto: bool = field(init=False, default=None)
    title_auto: bool = field(init=False, default=None)

    # backend objects
    ax: Any = field(init=False, repr=False, default=None)
    cb: Any = field(init=False, repr=False, default=None)
    legend: Any = field(init=False, repr=False, default=None)

    # parent figure object
    fig: Any = None

    # child plot objects
    plots: List[PlotTypes] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        # initialize axes backend
        self.init_backend()

    # backend methods
    # TODO: generalize the following methods for arbitrary backend
    def init_backend(self):
        with mpl.rc_context(fname=self.fig.fname, rc=self.fig.rc):
            self.ax = self.fig.fig.add_subplot(
                self.fig.nrows, self.fig.ncols, self.index
            )

    def clear_backend(self):
        for plot in self.plots:
            plot.clear()

        self.clear_colorbar()
        self.clear_legend()

        self.ax.clear()
        self.ax.remove()
        self.ax = None

    def reset_backend(self):
        self.clear_backend()
        self.init_backend()

    def update_backend(self):
        with mpl.rc_context(fname=self.fig.fname, rc=self.fig.rc):
            ax = self.ax

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
            plot.axes = self

        self.plots.append(plot)

        return plot

    def update_plot_options(self):
        if self.xlim_auto:
            self.xlim = (
                min([p._default_xlim()[0] for p in self.plots]),
                max([p._default_xlim()[1] for p in self.plots]),
            )

        if self.ylim_auto:
            self.ylim = (
                min([p._default_ylim()[0] for p in self.plots]),
                max([p._default_ylim()[1] for p in self.plots]),
            )

        if self.xlabel_auto:
            units = [p._xunits() for p in self.plots]
            xlabels = [p._default_xlabel(units=False) for p in self.plots]

            if len(set(units)) == 1 and len(set(xlabels)) == 1:
                self.xlabel = set(xlabels).pop()
                if set(units).pop():
                    self.xlabel += " [" + set(units).pop() + "]"
            elif len(set(units)) == 1:
                xlabels = [p._default_xlabel(units=False) for p in self.plots]
                if set(units).pop():
                    self.xlabel += " [" + set(units).pop() + "]"
            else:
                xlabels = [p._default_xlabel(units=True) for p in self.plots]
                self.xlabel = ", ".join(xlabels)

        if self.ylabel_auto:
            units = [p._yunits() for p in self.plots]
            ylabels = [p._default_ylabel(units=False) for p in self.plots]

            if len(set(units)) == 1 and len(set(ylabels)) == 1:
                self.ylabel = set(ylabels).pop()
                if set(units).pop():
                    self.ylabel += " [" + set(units).pop() + "]"
            elif len(set(units)) == 1:
                ylabels = [p._default_ylabel(units=False) for p in self.plots]
                self.ylabel = ", ".join(ylabels)
                if set(units).pop():
                    self.ylabel += " [" + set(units).pop() + "]"
            else:
                ylabels = [p._default_ylabel(units=True) for p in self.plots]
                self.ylabel = ", ".join(ylabels)

        if self.title_auto:
            titles = [p._default_title() for p in self.plots]

            if len(set(titles)) == 1:
                self.title = set(titles).pop()
            else:
                self.title = None

    def update(self):
        self.update_plot_options()
        self.update_backend()

        if len(self.plots) > 1:
            self.init_legend()

    def redo_plots(self):

        self.reset_backend()

        for plot in self.plots:
            plot.build_canvas()

        self.update()

    # legend methods
    def init_legend(self):
        if self.legend_show:
            handles, labels = self.ax.get_legend_handles_labels()
            with mpl.rc_context(fname=self.fig.fname, rc=self.fig.rc):
                # show legend
                self.legend = self.ax.legend(
                    handles=handles,
                    labels=labels,
                    loc=self.legend_loc,
                    frameon=self.legend_frameon,
                )

    def clear_legend(self):
        if self.legend:
            self.legend.remove()
            self.legend = None

    # colorbar methods
    def init_colorbar(self, plot: PlotTypes):
        if self.cb_show:
            with mpl.rc_context(fname=self.fig.fname, rc=self.fig.rc):
                # show colorbar
                self.cb = self.ax.get_figure().colorbar(plot.h, aspect=30)

                # set colorbar title)
                self.cb.set_label(label=plot.cb_title)

    def clear_colorbar(self):
        if self.cb:
            self.cb.remove()
            self.cb = None

    def copy(self):
        new = copy(self)

        for plot in new.plots:
            plot.axes = new

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
        ]
