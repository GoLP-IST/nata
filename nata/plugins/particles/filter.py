# -*- coding: utf-8 -*-
from typing import List
from typing import Optional

from nata.containers import ParticleDataset
from nata.plugins.register import register_container_plugin


@register_container_plugin(ParticleDataset, name="filter")
def filter_particle_dataset(
    dataset: ParticleDataset, quantities: Optional[List[str]] = None
) -> ParticleDataset:
    if quantities is not None:
        quants = {}
        for quant in quantities:
            quants[quant] = (
                dataset.quantities[quant]
                if quant in dataset.quantities.keys()
                else None
            )

    return ParticleDataset(
        iteration=dataset.axes["iteration"],
        time=dataset.axes["time"],
        name=dataset.name,
        quantities=quants if quantities is not None else dataset.quantities,
    )
