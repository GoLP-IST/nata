# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List
from typing import Optional

import numpy as np

from nata.plots.types import BasePlot


@dataclass
class LinePlot(BasePlot):
    """Line plot class.

    Parameters
    ----------
    ls: ``str``, optional
        Linestyle of the line. See
        :meth:`matplotlib.lines.Line2D.set_linestyle` for available options.

    lw: ``float``, optional
        Line width in points. If not provided, defaults to ``1``.

    color: ``str``, optional
        Color of the line. See :mod:`matplotlib.colors` for available options.

    alpha: ``float``, optional
        Line alpha value. Must be between ``0`` and ``1``.

    marker: ``str``, optional
        Marker to be used in defined line points. See :mod:`matplotlib.markers`
        for available options.

    ms: ``float``, optional
        Marker size in points.

    antialiased: ``bool``, optional
        Controls wether the plot should be antialiased. If not provided,
        defaults to ``True``.

    """

    ls: Optional[str] = None
    lw: Optional[float] = 1
    color: Optional[str] = None
    alpha: Optional[float] = None
    marker: Optional[str] = None
    ms: Optional[float] = None
    antialiased: Optional[bool] = True

    def _default_xlim(self):
        return (self.data.axes[0].min, self.data.axes[0].max)

    def _default_ylim(self):
        return (np.min(self.data.data), np.max(self.data.data))

    def _default_xlabel(self, units=True):
        return self.data.axes[0].get_label(units)

    def _default_ylabel(self, units=True):
        return self.data.get_label(units)

    def _default_title(self):
        return self.data.get_time_label()

    def _default_label(self):
        return self.data.get_label(units=False)

    def _xunits(self):
        return f"${self.data.axes[0].units}$" if self.data.axes[0].units else ""

    def _yunits(self):
        return f"${self.data.units}$" if self.data.units else ""

    def build_canvas(self):
        # get plot axes and data
        x = self.data.axes[0].data
        y = self.data.data

        # build plot
        self.h = self.axes.ax.plot(
            x,
            y,
            ls=self.ls,
            lw=self.lw,
            c=self.color,
            alpha=self.alpha,
            marker=self.marker,
            ms=self.ms,
            label=self.label,
            antialiased=self.antialiased,
        )

    @classmethod
    def style_attrs(cls) -> List[str]:
        return [
            "ls",
            "lw",
            "color",
            "alpha",
            "marker",
            "ms",
            "antialiased",
        ] + BasePlot.style_attrs()
