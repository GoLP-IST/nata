# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np
from numpy.lib import recfunctions

from nata.containers.utils import unstructured_to_structured


def test_unstructured_to_structured():
    arr = da.array([1, 2, 3])
    structured_dtype = np.dtype([("f1", int), ("f2", float), ("f3", np.bool8)])

    # expected is based on doing numpy 'unstructured_to_structured'
    expected = recfunctions.unstructured_to_structured(arr.compute(), structured_dtype)

    # do own 'unstructured_to_structured'
    converted_arr = unstructured_to_structured(arr, structured_dtype)

    assert isinstance(converted_arr, da.Array)
    np.testing.assert_array_equal(converted_arr, expected)
