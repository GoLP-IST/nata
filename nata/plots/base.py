import attr
import numpy as np
import matplotlib.pyplot as plt

class BasePlot:    
    def __init__(self, parent=None, show=True, **kwargs):

        self._parent = parent
        self._show = show
        self._plt = plt

        self.set_attrs(
            attr_list=[
                ("fontsize", 16),
                ("pad", 10),
                ("figsize", (9,6)),
                ("aspect", "auto")
            ],
            kwargs=kwargs            
        )            
        
        self.set_style(style="default")
    
    def set_attrs(self, attr_list, kwargs):
        for (attr, default) in attr_list:
            setattr(self, "_" + attr, kwargs.get(attr, default))
            setattr(self, "_" + attr + "_auto", attr not in kwargs)

    def set_style(self, style="default"):
        # TODO: Allow providing of a general style from arguments to BasePlot
        #       or from a style file

        self._plt.rcParams['xtick.major.pad'] = self._pad
        self._plt.rcParams['ytick.major.pad'] = self._pad

        self._plt.rcParams['text.usetex'] = True
        self._plt.rcParams['font.serif'] = 'Palatino'
        self._plt.rcParams['font.size'] = self._fontsize


    def show(self):
        if self._show:
            self._plt.show()


    # def update(self):

    #     if self._xlim_auto:
    #         self.xlim = (self._parent.xmin[0], self._parent.xmax[0])
        
    #     if self._ylim_auto:
    #         self.ylim = (self._parent.xmin[1], self._parent.xmax[1])

    #     if self._xlabel_auto:
    #         self._xlabel = self._parent._axes[0].get_label()

    #     if self._ylabel_auto:
    #         self._ylabel = self._parent._axes[1].get_label()

    #     if self._title_auto:
    #         self._title = self._parent.get_title()