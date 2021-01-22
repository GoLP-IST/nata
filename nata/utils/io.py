# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Iterator
from typing import Union


class FileList:
    def __init__(self, entrypoint: Union[str, Path], recursive: bool = True):
        self._entrypoint = (
            Path(entrypoint) if not isinstance(entrypoint, Path) else entrypoint
        )
        self._recursive = recursive

        if not self._entrypoint.exists():
            if not self._entrypoint.parent.exists():
                raise ValueError("Passed a non-existing path!")
            self._search_pattern = self._entrypoint.name
            self._entrypoint = self._entrypoint.parent.absolute()
        else:
            self._entrypoint = self._entrypoint.absolute()
            self._search_pattern = "*"

    @property
    def entrypoint(self) -> Path:
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
    def paths(self) -> Iterator[Path]:
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
    def parent_directory(self) -> Path:
        if self._entrypoint.is_dir():
            return self._entrypoint
        else:
            return self._entrypoint.parent
