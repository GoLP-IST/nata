def filter_kwargs(cls, **kwargs):
    
    # get selectable kwargs from class definition
    kwargs_sel = []
    for attr in cls.__dict__["__attrs_attrs__"]:
        if attr.init:
            kwargs_sel.append(attr.name)

    # build filtered kwargs
    kwargs_flt = {}
    for attr in kwargs_sel:
        try: 
            prop = kwargs.pop(attr)
            kwargs_flt[attr] = prop
        except:
            continue
        
    return kwargs_flt