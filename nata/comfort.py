# -*- coding: utf-8 -*-
import pathlib
from typing import Union

import attr

from nata.containers import DatasetCollection


@attr.s
class _FileList:
    _entrypoint: pathlib.Path = attr.ib(converter=pathlib.Path)
    _recursive: bool = attr.ib(default=True)
    _search_pattern: str = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        if not self._entrypoint.exists():
            if not self._entrypoint.parent.exists():
                raise ValueError("Passed a non-existing path!")
            self._search_pattern = self._entrypoint.name
            self._entrypoint = self._entrypoint.parent.absolute()
        else:
            self._entrypoint = self._entrypoint.absolute()
            self._search_pattern = "*"

    @property
    def entrypoint(self) -> pathlib.Path:
        return self._entrypoint

    @property
    def recursive(self) -> bool:
        return self._recursive

    @property
    def search_pattern(self) -> str:
        return self._search_pattern

    @property
    def is_single_file(self) -> bool:
        return self._entrypoint.is_file()

    @property
    def paths(self) -> pathlib.Path:
        if self.is_single_file:
            yield self._entrypoint

        if self._recursive:
            list_generator = self._entrypoint.rglob(self._search_pattern)
        else:
            list_generator = self._entrypoint.glob(self._search_pattern)

        for p in sorted(list_generator):
            if p.is_file():
                yield p

    @property
    def parent_directory(self) -> pathlib.Path:
        if self._entrypoint.is_dir():
            return self._entrypoint
        else:
            return self._entrypoint.parent


def load(
    path: Union[str, pathlib.Path], recursive: bool = True,
) -> DatasetCollection:
    """
    Lazy function for loading simulation data.

    Takes a path and tries to load all possible files into a container. If
    the path is a single file, only a container fitting for this file is
    returned. If the file is not a single file but rather a directory then
    diagnostic files are searched for recursively. You can control the search
    by providing a search pattern.

    Parameters
    ----------
    ``path`` : ``str``, ``pathlib.Path``

        Path which will be used to search for diagnostic files. If the path
        does not end on a file or a directory, the string after the last
        ``/`` will be used a search pattern.

    ``recursive`` : ``bool``, default: ``True``

        Parameter for deciding if a path should be traversed recursively
        ``True`` or if only the top directory should be searched ``False``.
        The default option is ``True``.

    Returns
    -------
    ``dataobj`` : ``GridDataset``, ``DatasetCollection``

        Returns specific container based what was provided as ``path``

    Examples
    --------
        # loads all diagnostics
        >>> data = nata.load("sim_dir/MS/")
        >>> type(data)
        nata.containers.DatasetCollection
        # e1 is a directory with only one file `e1.h5`
        >>> data = nata.load("sim_dir/MS/FLD/e1")
        >>> type(data)
        nata.containers.GridDataset
        # load data recursively by matching
        >>> data = nata.load("sim_dir/MS/e1*.h5")
    """
    filelist = _FileList(path, recursive=recursive)
    collection = DatasetCollection(root_path=filelist.parent_directory)

    for p in filelist.paths:
        collection.append(p)

    # TODO: add here a possibility to reduce the data objects fully
    #   - should return on Dataset object if only one object is present
    return collection


def activate_logging(loggin_level: str = "info"):
    import logging

    if loggin_level == "notset":
        level = logging.NOTSET
    elif loggin_level == "debug":
        level = logging.DEBUG
    elif loggin_level == "info":
        level = logging.INFO
    elif loggin_level == "warning":
        level = logging.WARNING
    elif loggin_level == "error":
        level = logging.ERROR
    elif loggin_level == "critical":
        level = logging.CRITICAL
    else:
        raise ValueError(
            "Invalid loggin level provided! "
            + "Allowed are 'notset', 'debug', 'info', 'warning', 'error', "
            + "and 'critical'!"
        )

    logging.basicConfig(format="%(levelname)s :: %(message)s", level=level)
