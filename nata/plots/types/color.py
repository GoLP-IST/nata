# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List
from typing import Optional

import matplotlib.colors as clr
import numpy as np

from nata.plots.types import BasePlot


@dataclass
class ColorPlot(BasePlot):
    """Color plot class.

    Parameters
    ----------
    vmin: ``float``, optional
        Minimum of the colorbar axis. If not  provided, it is inferred from
        the dataset represented in the plot.

    vmax: ``float``, optional
        Same as ``vmin`` for the maximum of the colorbar axis.

    cb_title: ``str``, optional
        Colorbar title. If not provided, it is inferred from the dataset
        represented in the plot.

    cb_scale: ``{'linear','log', 'symlog'}``, optional
        Scale of the colorbar. If not provided, defaults to ``'linear'``.

    cb_map: ``str``, optional
        Colormap used to represent the data. See
        :func:`matplotlib.pyplot.colormaps` for available options. If not
        provided, defaults to ``rainbow``.

    cb_linthresh: ``float``, optional
        Range within which the colorbar axis is linear. Applicable only when
        ``cb_scale`` is set to ``'symlog'``. If not provided, defaults to
        ``1e-5``.

    interpolation: ``str``, optional
        Interpolation method used. See :func:`matplotlib.pyplot.imshow` for
        available options. If not provided, defaults to ``none``.

    """

    vmin: Optional[float] = None
    vmax: Optional[float] = None
    cb_map: Optional[str] = "rainbow"
    cb_scale: Optional[str] = "linear"
    cb_linthresh: Optional[float] = 1e-5
    cb_title: Optional[str] = None
    interpolation: Optional[str] = "none"

    def __post_init__(self):
        if self.cb_title is None:
            self.cb_title = self.data.get_label()

        super().__post_init__()

    def _default_xlim(self):
        return (self.data.axes[0].min, self.data.axes[0].max)

    def _default_ylim(self):
        return (self.data.axes[1].min, self.data.axes[1].max)

    def _default_xlabel(self, units=True):
        return self.data.axes[0].get_label(units)

    def _default_ylabel(self, units=True):
        return self.data.axes[1].get_label(units)

    def _default_title(self):
        return self.data.get_time_label()

    def _default_label(self):
        return self.data.get_label(units=False)

    def _xunits(self):
        return f"${self.data.axes[0].units}$" if self.data.axes[0].units else ""

    def _yunits(self):
        return f"${self.data.axes[1].units}$" if self.data.axes[1].units else ""

    def build_canvas(self):
        # get plot extent and data
        extent = (
            self.data.axes[0].min,
            self.data.axes[0].max,
            self.data.axes[1].min,
            self.data.axes[1].max,
        )
        z = np.transpose(self.data.data)

        # build color map norm
        if self.cb_scale == "log":
            # convert values to positive numbers
            z = np.abs(z) + 1e-16

            # adjust min and max values
            # TODO: do this only if vmin was not init
            # self.vmin = np.min(z)

            # if self.vmax_auto:
            # self.vmax = np.max(z)

            # set color map norm
            self.cb_norm = clr.LogNorm(vmin=self.vmin, vmax=self.vmax)
        elif self.cb_scale == "symlog":
            # set color map norm
            self.cb_norm = clr.SymLogNorm(
                vmin=self.vmin,
                vmax=self.vmax,
                linthresh=self.cb_linthresh,
                base=10,
            )
        else:
            self.cb_norm = clr.Normalize(vmin=self.vmin, vmax=self.vmax)

        # build plot
        self.h = self.axes.ax.imshow(
            z,
            extent=extent,
            origin="lower",
            cmap=self.cb_map,
            norm=self.cb_norm,
            interpolation=self.interpolation,
        )

        self.axes.init_colorbar(plot=self)

    @classmethod
    def style_attrs(cls) -> List[str]:
        return [
            "vmin",
            "vmax",
            "cb_map",
            "cb_scale",
            "cb_linthresh",
            "cb_title",
            "interpolation",
        ] + BasePlot.style_attrs()
