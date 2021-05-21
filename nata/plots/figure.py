# -*- coding: utf-8 -*-
from typing import Optional
from typing import Sequence
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .elements import Scale
from .elements import Theme
from .elements import Ticks
from .elements import mpl_norm_from_scale
from .elements import mpl_scale_from_scale

Numbers = Union[int, float]


class Figure:
    def __init__(
        self,
        xrange: Optional[Sequence[Numbers]] = None,
        yrange: Optional[Sequence[Numbers]] = None,
        xscale: Optional[Scale] = None,
        yscale: Optional[Scale] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xticks: Optional[Ticks] = None,
        yticks: Optional[Ticks] = None,
        title: Optional[str] = None,
        aspect: Optional[Union[str, Numbers]] = None,
        size: Optional[Sequence[Numbers]] = None,
        theme: Theme = Theme(name="light"),
    ):

        self.theme = theme

        with mpl.rc_context(rc=self.theme.rc):
            self.backend_fig = plt.figure()
            self.backend_ax = self.backend_fig.add_subplot()

        self.xrange = xrange
        self.yrange = yrange
        self.xscale = xscale
        self.yscale = yscale
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xticks = xticks
        self.yticks = yticks
        self.title = title
        self.size = size
        self.aspect = aspect

    def line(self, x, y, color=None, style=None, width=None, alpha=None):
        with mpl.rc_context(rc=self.theme.rc):
            self.backend_ax.plot(
                x,
                y,
                linestyle=style,
                linewidth=width,
                color=color,
                alpha=alpha,
            )

        if not self.xrange:
            self.xrange = (np.min(x), np.max(x))

        # if not self.yrange:
        #     self.yrange = (np.min(y), np.max(y))

    def scatter(
        self,
        x,
        y,
        color=None,
        colorrange=None,
        colorscale=None,
        colormap=None,
        colorbar=None,
        style=None,
        size=None,
        alpha=None,
    ):

        has_colors = (
            color is not None
            and isinstance(color, (Sequence, np.ndarray))
            and len(color) == len(x)
        )

        has_single_color = color is not None and (
            isinstance(color, str)
            or (isinstance(color, Sequence) and len(color) in (3, 4))
        )

        with mpl.rc_context(rc=self.theme.rc):
            sct = self.backend_ax.scatter(
                x,
                y,
                marker=style,
                s=size,
                c=color if has_colors else None,
                color=color if has_single_color else None,
                norm=mpl_norm_from_scale(
                    colorscale,
                    (
                        colorrange[0] if colorrange else np.min(color),
                        colorrange[1] if colorrange else np.max(color),
                    ),
                )
                if has_colors
                else None,
                cmap=colormap,
                alpha=alpha,
            )

            if has_colors and colorbar and colorbar.visible:
                cb = self.backend_fig.colorbar(
                    sct, ax=self.backend_ax, label=colorbar.label
                )

                if colorbar.ticks and colorbar.ticks.values:
                    cb.set_ticks(colorbar.ticks.values)

                if colorbar.ticks and colorbar.ticks.labels:
                    cb.set_ticklabels(colorbar.ticks.labels)

    def image(
        self,
        x,
        y,
        color,
        colorrange=None,
        colorscale=None,
        colormap=None,
        colorbar=None,
    ):

        with mpl.rc_context(rc=self.theme.rc):
            img = self.backend_ax.imshow(
                np.transpose(color),
                extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
                origin="lower",
                norm=mpl_norm_from_scale(
                    colorscale,
                    (
                        colorrange[0] if colorrange else np.min(color),
                        colorrange[1] if colorrange else np.max(color),
                    ),
                ),
                cmap=colormap,
                aspect=self.aspect,
            )

            if colorbar and colorbar.visible:
                cb = self.backend_fig.colorbar(
                    img, ax=self.backend_ax, label=colorbar.label
                )

                if colorbar.ticks and colorbar.ticks.values:
                    cb.set_ticks(colorbar.ticks.values)

                if colorbar.ticks and colorbar.ticks.labels:
                    cb.set_ticklabels(colorbar.ticks.labels)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = size
        if size:
            self.backend_fig.set_size_inches(size[0], size[1])

    @property
    def aspect(self):
        return self._aspect

    @aspect.setter
    def aspect(self, aspect):
        self._aspect = aspect
        if aspect:
            self.backend_ax.set_aspect(aspect)

    @property
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, theme):
        self._theme = theme

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

        if title:
            self.backend_ax.set_title(title)

    @property
    def xlabel(self):
        return self._xlabel

    @xlabel.setter
    def xlabel(self, label):
        self._xlabel = label
        if label:
            self.backend_ax.set_xlabel(label)

    @property
    def ylabel(self):
        return self._ylabel

    @ylabel.setter
    def ylabel(self, label):
        self._ylabel = label
        if label:
            self.backend_ax.set_ylabel(label)

    @property
    def xscale(self):
        return self._xscale

    @xscale.setter
    def xscale(self, scale):
        self._xscale = scale

        if isinstance(scale, Scale):
            self.backend_ax.set_xscale(mpl_scale_from_scale(scale))

    @property
    def yscale(self):
        return self._yscale

    @yscale.setter
    def yscale(self, scale):
        self._yscale = scale

        if isinstance(scale, Scale):
            self.backend_ax.set_yscale(mpl_scale_from_scale(scale))

    @property
    def xrange(self):
        return self._xrange

    @xrange.setter
    def xrange(self, xrange):
        self._xrange = xrange

        if xrange:
            self.backend_ax.set_xlim(xrange)

    @property
    def yrange(self):
        return self._yrange

    @yrange.setter
    def yrange(self, yrange):
        self._yrange = yrange

        if yrange:
            self.backend_ax.set_ylim(yrange)

    @property
    def xticks(self):
        return self._xticks

    @xticks.setter
    def xticks(self, ticks):
        self._xticks = ticks

        if ticks and ticks.values:
            self.backend_ax.set_xticks(ticks.values)
        if ticks and ticks.labels:
            self.backend_ax.set_xticklabels(ticks.labels)

    @property
    def yticks(self):
        return self._yticks

    @yticks.setter
    def yticks(self, ticks):
        self._yticks = ticks

        if ticks and ticks.values:
            self.backend_ax.set_yticks(ticks.values)
        if ticks and ticks.labels:
            self.backend_ax.set_yticklabels(ticks.labels)

    def close(self):
        plt.close(self.backend_fig)

    def show(self):
        with mpl.rc_context(rc=self.theme.rc):
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = self.backend_fig
            plt.show()

    def save(self, name: str, dpi: int = 150):
        with mpl.rc_context(rc=self.theme.rc):
            self.backend_fig.savefig(name, dpi=dpi)

    def _repr_html_(self):
        self.show()
