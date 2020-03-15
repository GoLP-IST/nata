# -*- coding: utf-8 -*-
import numpy as np
from hypothesis import assume
from hypothesis.extra.numpy import array_shapes
from hypothesis.extra.numpy import arrays
from hypothesis.extra.numpy import basic_indices
from hypothesis.extra.numpy import complex_number_dtypes
from hypothesis.extra.numpy import floating_dtypes
from hypothesis.extra.numpy import integer_dtypes
from hypothesis.strategies import complex_numbers
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import none
from hypothesis.strategies import one_of


@composite
def intc_bounded_intergers(draw):
    return draw(
        integers(
            min_value=np.iinfo(np.intc).min, max_value=np.iinfo(np.intc).max
        ),
    )


@composite
def number_or_none(draw):
    return draw(
        one_of(
            none(),
            floats(allow_nan=False),
            complex_numbers(allow_nan=False),
            intc_bounded_intergers(),
        )
    )


@composite
def number(draw, include_complex_numbers=True):
    if include_complex_numbers:
        return draw(
            one_of(
                floats(allow_nan=False, allow_infinity=False, width=16),
                complex_numbers(allow_nan=False, allow_infinity=False),
                intc_bounded_intergers(),
            )
        )
    else:
        return draw(
            one_of(
                floats(allow_nan=False, allow_infinity=False, width=16),
                intc_bounded_intergers(),
            )
        )


@composite
def anyarray(draw, min_dims=0, max_dims=2):
    arr = draw(
        arrays(
            dtype=one_of(
                integer_dtypes(), floating_dtypes(), complex_number_dtypes()
            ),
            shape=array_shapes(min_dims=min_dims, max_dims=max_dims),
            unique=True,
        )
    )
    assume(not np.any(np.isnan(arr)))
    assume(np.all(np.isfinite(arr)))

    return arr


@composite
def array_and_basic_indices(draw, array_min_dims=0, array_max_dims=2):
    arr = draw(anyarray(min_dims=array_min_dims, max_dims=array_max_dims))
    ind = draw(basic_indices(arr.shape))
    return arr, ind
