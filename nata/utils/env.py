# -*- coding: utf-8 -*-
def inside_ipython():
    try:
        get_ipython()
        inside = True
    except NameError:
        inside = False

    return inside


def inside_notebook():
    if inside_ipython():
        shell = get_ipython().__class__.__name__  # noqa: F821
        if shell == "ZMQInteractiveShell":
            return True

    return False


def inside_ipython_terminal():
    if inside_ipython():
        shell = get_ipython().__class__.__name__  # noqa: F821
        if shell == "TerminalInteractiveShell":
            return True

    return False
