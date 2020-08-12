# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import List

import numpy as np

from nata.containers import ParticleDataset
from nata.containers import ParticleQuantity
from nata.plugins.register import register_container_plugin


@register_container_plugin(ParticleDataset, name="filter")
def filter_particle_dataset(
    dataset: ParticleDataset,
    mask: List[bool] = None,
    quantities: List[str] = None,
    slicing: slice = None,
) -> ParticleDataset:
    """Filters a :class:`nata.containers.ParticleDataset` according to a\
       selection of quantities.

        Parameters
        ----------
        mask: ``np.ndarray``, optional
            Array of booleans indicating the particles to be filtered. Particles
            with ``True`` (``False``) mask entries are selected (hidden). The
            shape of ``mask`` must match that of each particle quantity.

        slicing: ``slice``, optional
            Slice of particles to be filtered. Acts only on particle indices
            and not on time, as time slicing should be done on the dataset.
            When provided together with the ``mask argument``, slicing is done
            on the masked array.

        quantities: ``list``, optional
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

    dataset_c = deepcopy(dataset)

    if quantities is not None:
        if not isinstance(quantities, list) or not all(
            isinstance(quant, str) for quant in quantities
        ):
            raise TypeError(f"'quantities' must be a list of strings.")

        quants = {
            quant: dataset_c.quantities[quant]
            for quant in quantities
            if quant in dataset_c.quantities.keys()
        }

        if not bool(quants):
            raise ValueError(
                f"None of the quantities in 'quantities' were found in the "
                + "dataset."
            )

    else:
        quants = dataset_c.quantities

    if mask is not None:
        if not isinstance(mask, np.ndarray) or not mask.dtype == bool:
            raise TypeError(f"'mask' must be an 'np.ndarray' of dtype 'bool'.")

        quant = next(iter(quants.values()))
        if quant.data.shape != mask.shape:
            raise ValueError(
                f"'mask.shape' ({mask.shape}) does not match the shape of "
                + "each quantity ({quant.data.shape})."
            )

        for name, quant in quants.items():
            quants[name].data[~mask] = np.ma.masked

    if slicing is not None:
        if not isinstance(slicing, slice):
            raise TypeError(f"'slicing' must be of type 'slice'.")

        if len(dataset_c) > 1:
            slicing = (slice(None), slicing)

        quants = {
            quant.name: ParticleQuantity(
                data=quant.data[slicing],
                name=quant.name,
                label=quant.label,
                unit=quant.unit,
                dtype=quant.data.dtype,
            )
            for quant in quants.values()
        }

    return ParticleDataset(
        iteration=dataset_c.axes["iteration"],
        time=dataset_c.axes["time"],
        name=dataset_c.name,
        quantities=quants,
    )
