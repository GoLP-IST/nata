# -*- coding: utf-8 -*-
from typing import List

from nata.containers import ParticleDataset
from nata.plugins.register import register_container_plugin


@register_container_plugin(ParticleDataset, name="filter")
def filter_particle_dataset(
    dataset: ParticleDataset, quantities: List[str]
) -> ParticleDataset:
    """Filters a :class:`nata.containers.ParticleDataset` according to a\
       selection of quantities.

        Parameters
        ----------
        quantities: ``list``
            List of quantities to be filtered, ordered by the way they should be
            sorted in the returned dataset.

        Returns
        ------
        :class:`nata.containers.ParticleDataset`:
            Filtered dataset with only the quantities selected in
            ``quantities``.

        Examples
        --------
        The filter plugin is used to get dataset with only a selection, say
        ``'x1'`` and ``'x2'``, of its quantities.

        >>> from nata.containers import ParticleDataset
        >>> ds = ParticleDataset("path/to"file")
        >>> ds_flt = ds.filter(quantities=["x1","p1"])

    """

    if quantities is not None:
        quants = {}
        for quant in quantities:
            quants[quant] = (
                dataset.quantities[quant]
                if quant in dataset.quantities.keys()
                else None
            )
    else:
        raise ValueError("")

    return ParticleDataset(
        iteration=dataset.axes["iteration"],
        time=dataset.axes["time"],
        name=dataset.name,
        quantities=quants,
    )
