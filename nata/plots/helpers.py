# -*- coding: utf-8 -*-
def filter_style(cls, style: dict = dict(),) -> dict:

    style_sel = cls.style_attrs()
    # build filtered style attributes
    style_flt = dict()
    for attrib in style_sel:
        try:
            # TODO: make this use defaults - most likely it can be None @fabio
            #       - if so, try and except can be removed
            #       - else KeyError should be excepted
            prop = style.pop(attrib)
            style_flt[attrib] = prop
        except:  # noqa: E722
            continue

    return style_flt
