from abc import ABC, abstractmethod

import attr
from attr.validators import optional, instance_of

import numpy as np

from nata.plots.data import PlotData

@attr.s
class BasePlot(ABC):

    # parent axes object
    _axes = attr.ib(
        init=True, 
        repr=False
    )
    
    # data object
    _data: PlotData = attr.ib(init=True, repr=False)

    # plot handle object
    _h: attr.ib(init=False, repr=False)

    label: str = attr.ib(
        default=None, 
        validator=optional(instance_of(str))
    )
    
    # validator for parent axes object
    @_axes.validator
    def _axes_validator(self, attr, _axes):
        
        if _axes.xlim is None:
            _axes.xlim = self._default_xlim()

        if _axes.ylim is None:
            _axes.ylim = self._default_ylim()

        if _axes.xlabel is None:
            _axes.xlabel = self._default_xlabel()

        if _axes.ylabel is None:
            _axes.ylabel = self._default_ylabel()

        if _axes.title is None:
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

    def _default_xlabel(self):
        return ""

    def _default_ylabel(self):
        return ""

    def _default_title(self):
        return ""

    def _default_label(self):
        return ""
        
    def __attrs_post_init__(self):
        
        self.build_canvas()

    def build_canvas(self):
        pass

    def clear(self):
        pass