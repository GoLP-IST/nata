# -*- coding: utf-8 -*-
from abc import ABC

import attr
from attr.validators import instance_of
from attr.validators import optional

from nata.plots.data import PlotData


@attr.s
class BasePlot(ABC):

    # parent axes object
    _axes = attr.ib(init=True, repr=False)

    # data object
    _data: PlotData = attr.ib(init=True, repr=False)

    # plot handle object
    _h: attr.ib(init=False, repr=False)

    label: str = attr.ib(default=None, validator=optional(instance_of(str)))

    # validator for parent axes object
    @_axes.validator
    def _axes_validator(self, attr, _axes):

        if _axes.xlim is None:
            _axes._xlim_auto = True
            _axes.xlim = self._default_xlim()

        if _axes.ylim is None:
            _axes._ylim_auto = True
            _axes.ylim = self._default_ylim()

        if _axes.xlabel is None:
            _axes._xlabel_auto = True
            _axes.xlabel = self._default_xlabel()

        if _axes.ylabel is None:
            _axes._ylabel_auto = True
            _axes.ylabel = self._default_ylabel()

        if _axes.title is None:
            _axes._title_auto = True
            _axes.title = self._default_title()

    # validator for label
    @label.validator
    def label_validator(self, attr, _axes):

        if self.label is None:
            self.label = self._default_label()

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

    def __attrs_post_init__(self):

        self.build_canvas()

    def build_canvas(self):
        pass

    def clear(self):
        pass
