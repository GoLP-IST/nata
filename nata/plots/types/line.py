# -*- coding: utf-8 -*-
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from nata.plots.data import PlotData
from nata.plots.types import BasePlot


class LinePlot(BasePlot):
    def __init__(
        self,
        axes,
        data: PlotData,
        label: Optional[str] = None,
        ls: Optional[str] = None,
        lw: Optional[Union[float, int]] = None,
        c: Optional[str] = None,
        alpha: Optional[Union[float, int]] = None,
        marker: Optional[str] = None,
        ms: Optional[Union[float, int]] = None,
        antialiased: Optional[bool] = True,
    ):

        self.ls = ls
        self.lw = lw
        self.c = c
        self.alpha = alpha
        self.marker = marker
        self.ms = ms
        self.antialiased = antialiased

        super().__init__(axes, data, label)

    def _default_xlim(self):
        return (self._data.axes[0].min, self._data.axes[0].max)

    def _default_ylim(self):
        return (np.min(self._data.data), np.max(self._data.data))

    def _default_xlabel(self, units=True):
        return self._data.axes[0].get_label(units)

    def _default_ylabel(self, units=True):
        return self._data.get_label(units)

    def _default_title(self):
        return self._data.get_time_label()

    def _default_label(self):
        return self._data.get_label(units=False)

    def _xunits(self):
        return f"${self._data.axes[0].units}$"

    def _yunits(self):
        return f"${self._data.units}$"

    def build_canvas(self):
        # get plot axes and data
        x = self._data.axes[0].data
        y = self._data.data

        # build plot
        self._h = self._axes._ax.plot(
            x,
            y,
            ls=self.ls,
            lw=self.lw,
            c=self.c,
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
            "c",
            "alpha",
            "marker",
            "ms",
            "antialiased",
        ] + BasePlot.style_attrs()
