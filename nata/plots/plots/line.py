# -*- coding: utf-8 -*-
import attr
import numpy as np
from attr.validators import instance_of
from attr.validators import optional

from nata.plots.plots.base import BasePlot


@attr.s
class LinePlot(BasePlot):

    # line plot options
    ls: str = attr.ib(default=None, validator=optional(instance_of(str)))
    lw: float = attr.ib(
        default=None, validator=optional(instance_of((int, float)))
    )
    c: str = attr.ib(default=None, validator=optional(instance_of(str)))
    alpha: float = attr.ib(
        default=None, validator=optional(instance_of((int, float)))
    )
    marker: str = attr.ib(default=None, validator=optional(instance_of(str)))
    ms: float = attr.ib(
        default=None, validator=optional(instance_of((int, float)))
    )
    antialiased: bool = attr.ib(
        default=True, validator=optional(instance_of(bool))
    )

    def _default_xlim(self):
        return (self._data.axes[0].min, self._data.axes[0].max)

    def _default_ylim(self):
        return (np.min(self._data.values), np.max(self._data.values))

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
        x = self._data.axes[0].values
        y = np.transpose(self._data.values)

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

        self._axes.update()
