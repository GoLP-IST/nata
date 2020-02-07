# -*- coding: utf-8 -*-
import attr

from nata.plots.base import BasePlot


@attr.s
class ParticlePlot1D(BasePlot):
    def _default_xlim(self):
        return (self.axes[0].min, self.axes[0].max)

    def _default_ylim(self):
        return (self.axes[1].min, self.axes[1].max)

    def _default_xlabel(self):
        return self.axes[0].get_label()

    def _default_ylabel(self):
        return self.axes[1].get_label()

    def _default_title(self):
        return self.data.get_time_label()

    def __attrs_post_init__(self):

        self.build_canvas()

    def build_canvas(self):

        # get plotting backend
        # plt = self.fig._plt

        # get figure
        fig = self.fig._fig

        ax = fig.add_subplot(111)

        # get plot axes and data
        x = self.data.values[0]
        y = self.data.values[1]

        # build plot
        ax.plot(x, y, ",")

        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        # set axes labels
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        # set title
        ax.set_title(label=self.title)

        # set aspect ratio
        ax.set_aspect(self.aspect)

        self._ax = ax
