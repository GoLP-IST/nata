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
def bounded_intergers(draw, min_value=None, max_value=None):
    lower_limit = (
        min_value
        if (min_value and (min_value > np.iinfo(np.intc).min))
        else np.iinfo(np.intc).min
    )
    upper_limit = (
        max_value
        if (max_value and (max_value < np.iinfo(np.intc).max))
        else np.iinfo(np.intc).max
    )

    return draw(integers(min_value=lower_limit, max_value=upper_limit),)


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


@composite
def array_with_two_entries(draw, array_length=10_000):
    length = draw(integers(1, max_value=array_length))
    arr = draw(
        arrays(
            dtype=one_of(integer_dtypes(), floating_dtypes()),
            shape=(length, 2),
        )
    )
    assume(not np.any(np.isnan(arr)))
    assume(np.all(np.isfinite(arr)))
    return arr
