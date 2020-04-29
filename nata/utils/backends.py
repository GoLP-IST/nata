# -*- coding: utf-8 -*-
from typing import MutableSequence
from typing import Optional
from typing import Sequence


def sort_particle_quantities(
    names: MutableSequence[str], order: Optional[Sequence[str]] = None
):
    if order is None:
        return sorted(names)
    else:
        sorted_ = []
        for key in order:
            filtered_names = filter(lambda elem: elem.startswith(key), names)
            for s in sorted(filtered_names):
                sorted_.append(s)
                names.remove(s)

        return sorted_ + sorted(names)
