# -*- coding: utf-8 -*-
from copy import copy
from dataclasses import dataclass
from dataclasses import field
from math import ceil
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from pkg_resources import resource_filename

from nata.plots.axes import Axes


@dataclass
class Figure:
    """Container of parameters and child objects (including plotting\
    backend-related objects) relevant to draw a figure.

    Parameters
    ----------
    figsize: ``tuple`` of ``float``, optional
        Tuple containing the width and height of the figure canvas in
        inches. If not provided, defaults to ``(6,4)``.

    nrows: ``int``, optional
        Number of rows available for figure axes. If not provided, defaults
        to ``1``.

    ncols: ``int``, optional
        Number of columns available for figure axes. If not provided,
        defaults to ``1``.

    style: ``{'light', 'dark'}``, optional
        Selection of standard nata style. If not provided, defaults to
        ``'light'``.

    fname: ``str``, optional
        Path to file with custom plotting backend parameters.

    rc: ``dict``, optional
        Dictionary with custom plotting backend parameters. Overrides
        parameters given in ``fname``.

    """

    figsize: Optional[Tuple[float]] = None
    nrows: Optional[int] = 1
    ncols: Optional[int] = 1
    style: Optional[str] = "light"
    fname: Optional[str] = None
    rc: Optional[Dict[str, Any]] = None

    # backend objects
    fig: Any = field(init=False, repr=False, default=None)

    # child axes objects
    _axes: List[Axes] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        # set plotting style
        self.set_style()

        # open figure object
        self.open()

    def set_style(self):
        if not self.fname:
            self.fname = resource_filename(
                __name__, "styles/" + self.style + ".rc"
            )

    # TODO: generalize methods for arbitrary backend
    def open(self):
        with mpl.rc_context(fname=self.fname, rc=self.rc):
            self.fig = plt.figure(figsize=self.figsize)

            if self.figsize is None:
                size = self.fig.get_size_inches()
                self.fig.set_size_inches(
                    size[0] * self.ncols, size[1] * self.nrows
                )

    def close(self):
        plt.close(self.fig)

    def reset(self):
        self.close()
        self.open()

    def show(self):
        """Shows the figure."""

        with mpl.rc_context(fname=self.fname, rc=self.rc):
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = self.fig

            self.fig.tight_layout()
            plt.show()

    def _repr_html_(self):
        """Calls :meth:`nata.plots.Figure.show`."""
        self.show()

    # TODO: generalize this for arbitrary backend
    def save(
        self, path, format: Optional[str] = None, dpi: Optional[float] = 150
    ):
        """Saves the figure to a file.

        Parameters
        ----------
            path: ``tuple`` of ``float``, optional
                Path in which to store the file.

            format: ``str``, optional
                File format, e.g. ``'png'``, ``'pdf'``, ``'svg'``. If not
                provided, the output format is inferred from the extension of
                ``path``.

            dpi: ``float``, optional
                Resolution in dots per inch. If not provided, defaults to
                ``150``.

        """
        with mpl.rc_context(fname=self.fname, rc=self.rc):
            self.fig.savefig(path, dpi=dpi, bbox_inches="tight")

    def copy(self):

        self.close()

        new = copy(self)
        new.open()

        for axes in new._axes:
            axes.fig = new

        return new

    def add_axes(self, style=dict()):

        new_index = len(self._axes) + 1

        if new_index > (self.nrows * self.ncols):
            # increase number of rows
            # TODO: really?
            self.nrows += 1

            if self.figsize is None:
                size = self.fig.get_size_inches()
                self.fig.set_size_inches(
                    size[0], size[1] * self.nrows / (self.nrows - 1)
                )

            for axes in self._axes:
                axes.redo_plots()

        axes = Axes(fig=self, index=new_index, **style)
        self._axes.append(axes)

        return axes

    def __mul__(self, other):
        """Combines two figures into one by superimposing the plots in axes with
        matching indices.
        """

        new = copy(self)

        for key, axes in new.axes.items():

            if key in other.axes:
                for plot in other.axes[key].plots:
                    axes.add_plot(plot=plot)

            axes.redo_plots()

        new.close()

        return new

    def __add__(self, other):
        """Combines two figures into one by adding new axes."""

        new = self.copy()

        new.nrows = ceil((len(new._axes) + len(other._axes)) / new.ncols)

        for axes in new._axes:
            axes.redo_plots()

        for axes in other._axes:
            # get a copy of old axes
            new_axes = axes.copy()

            # reset parent figure object
            new_axes.fig = new

            # redo plots in new axes
            new_axes.index = len(new._axes) + 1
            new_axes.redo_plots()

            # add axes to new list
            new._axes.append(new_axes)

        new.close()

        return new

    @property
    def axes(self) -> dict:
        """Dictionary of child `nata.plots.Axes` objects, where the key
        to each axes is its ``index`` property
        """
        return {axes.index: axes for axes in self._axes}

    @classmethod
    def style_attrs(self) -> List[str]:
        return [
            "figsize",
            "nrows",
            "ncols",
            "style",
            "fname",
            "rc",
        ]
