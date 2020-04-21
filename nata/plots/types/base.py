# -*- coding: utf-8 -*-
from typing import List
from typing import Optional

from nata.plots.data import PlotData


class BasePlot:
    def __init__(
        self, axes, data: PlotData, label: Optional[str] = None,
    ):
        # set child data object
        self._data = data

        # set defaults on parent axes object
        if axes.xlim is None:
            axes.xlim_auto = True
            axes.xlim = self._default_xlim()

        if axes.ylim is None:
            axes.ylim_auto = True
            axes.ylim = self._default_ylim()

        if axes.xlabel is None:
            axes.xlabel_auto = True
            axes.xlabel = self._default_xlabel()

        if axes.ylabel is None:
            axes.ylabel_auto = True
            axes.ylabel = self._default_ylabel()

        if axes.title is None:
            axes.title_auto = True
            axes.title = self._default_title()

        self._axes = axes

        # set default label
        if label is None:
            label = self._default_label()

        self.label = label

        self.build_canvas()

    def _default_xlim(self):
        return ()

    def _default_ylim(self):
        return ()

    def _default_xlabel(self, units=True):
        return ""

    def _default_ylabel(self, units=True):
        return ""

    def _default_title(self):
        return ""

    def _default_label(self):
        return ""

    def _xunits(self):
        return ""

    def _yunits(self):
        return ""

    def build_canvas(self):
        pass

    def clear(self):
        pass

    @classmethod
    def style_attrs(self) -> List[str]:
        return [
            "label",
        ]
