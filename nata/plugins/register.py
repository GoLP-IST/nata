# -*- coding: utf-8 -*-
from functools import wraps
from inspect import getfullargspec
from inspect import isclass
from inspect import isfunction
from typing import Optional


def _parse_register_args(callable_or_container, **kwargs):
    args = {"container": None, "name": None, "function": None}
    args.update({key: value for key, value in kwargs.items() if args.get(key) != value})

    if isfunction(callable_or_container):
        arg_spec = getfullargspec(callable_or_container)
        args["function"] = callable_or_container

        if arg_spec.args and (arg_spec.args[0] in arg_spec.annotations):
            args["container"] = arg_spec.annotations[arg_spec.args[0]]

    if isclass(callable_or_container):
        args["container"] = callable_or_container

    if args["container"] is not None:
        return args

    raise ValueError("No container to register plugin")


def register_container_plugin(
    callable_or_container=None,
    container=None,
    name: Optional[str] = None,
    plugin_type: str = "method",
):
    """Decorator for registering a plugin for a container."""
    arguments = _parse_register_args(
        callable_or_container, container=container, name=name
    )

    container = arguments.get("container")
    function_name = arguments.get("name")
    function = arguments.get("function")

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        plugin_name = function_name if function_name else func.__name__
        if plugin_type == "method":
            container.add_method_plugin(plugin_name, wrapper)
        elif plugin_type == "property":
            container.add_property_plugin(plugin_name, wrapper)
        else:
            raise ValueError("'plugin_type' has to be either 'property' or 'method'")
        return func

    if function:
        return decorator(function)

    return decorator


# TODO: following example breaks currently -- bug OR feature :)
# @register_container_plugin(name="not_so_different")
# def really_some_different_function(dataset: nata.GridDataset):
#     print(f"inside `really_some_different_function` -> {dataset.name}")
