# -*- coding: utf-8 -*-
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import List
from typing import Optional

from nata.plots.data import PlotData

# from nata.plots.axes import Axes


@dataclass
class BasePlot:
    """Base class for plot types.

    Parameters
    ----------
    label: ``str``, optional
        Label of plot, used to identify the plot in the parent
        :class:`nata.plots.Axes` object legend. If not provided, it is inferred
        from the child dataset object.

    """

    # style properties
    label: Optional[str] = None

    # plot data object
    data: PlotData = None

    # parent axes object
    axes: Any = None

    # backend objects
    h: Any = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # set defaults on parent axes object
        axes = self.axes

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

        # set default label
        if self.label is None:
            self.label = self._default_label()

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
