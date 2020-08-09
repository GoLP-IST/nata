# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nata.containers import _separation_newaxis

_TestCases_seperation_newaxis = {}
_TestCases_seperation_newaxis[":"] = (
    # key
    np.index_exp[:],
    # expected_deconstruction
    np.index_exp[:],
    # expected_temporal_key
    (),
    # expected_spatial_key
    (),
    # two_types
    True,
)
_TestCases_seperation_newaxis[":, :, :"] = (
    # key
    np.index_exp[:, :, :],
    # expected_deconstruction
    np.index_exp[:, :, :],
    # expected_temporal_key
    (),
    # expected_spatial_key
    (),
    # two_types
    True,
)
_TestCases_seperation_newaxis["int, int, int"] = (
    # key
    np.index_exp[4, 2, 0],
    # expected_deconstruction
    np.index_exp[4, 2, 0],
    # expected_temporal_key
    (),
    # expected_spatial_key
    (),
    # two_types
    True,
)
_TestCases_seperation_newaxis["newaxis, :"] = (
    # key
    np.index_exp[None, :],
    # expected_deconstruction
    np.index_exp[:],
    # expected_temporal_key
    (None,),
    # expected_spatial_key
    (),
    # two_types
    True,
)
_TestCases_seperation_newaxis["newaxis, newaxis, newaxis, :"] = (
    # key
    np.index_exp[None, None, None, :],
    # expected_deconstruction
    np.index_exp[:],
    # expected_temporal_key
    (None, None, None),
    # expected_spatial_key
    (),
    # two_types
    True,
)
_TestCases_seperation_newaxis[":, newaxis, :, newaxis, :, :, newaxis"] = (
    # key
    np.index_exp[:, None, :, None, :, :, None],
    # expected_deconstruction
    np.index_exp[:, :, :, :],
    # expected_temporal_key
    (),
    # expected_spatial_key
    (1, 3, 6),
    # two_types
    True,
)
_TestCases_seperation_newaxis[
    "newaxis, newaxis, :, newaxis, :, newaxis, :, :, newaxis"
] = (
    # key
    np.index_exp[None, None, :, None, :, None, :, :, None],
    # expected_deconstruction
    np.index_exp[:, :, :, :],
    # expected_temporal_key
    (None, None),
    # expected_spatial_key
    (1, 3, 6),
    # two_types
    True,
)
_TestCases_seperation_newaxis[
    "newaxis, newaxis, :, newaxis, :, newaxis, :, :, newaxis | single_type"
] = (
    # key
    np.index_exp[None, None, :, None, :, None, :, :, None],
    # expected_deconstruction
    np.index_exp[:, :, :, :],
    # expected_temporal_key
    (),
    # expected_spatial_key
    (0, 1, 3, 5, 8),
    # two_types
    False,
)


@pytest.mark.parametrize(
    [
        "key",
        "expected_deconstruction",
        "expected_temporal_key",
        "expected_spatial_key",
        "two_types",
    ],
    [args for args in _TestCases_seperation_newaxis.values()],
    ids=[names for names in _TestCases_seperation_newaxis.keys()],
)
def test_separation_newaxis(
    key,
    expected_deconstruction,
    expected_temporal_key,
    expected_spatial_key,
    two_types,
):
    deconstructed_key, temporal_key, spatial_key = _separation_newaxis(
        key, two_types=two_types
    )

    assert deconstructed_key == expected_deconstruction
    assert temporal_key == expected_temporal_key
    assert spatial_key == expected_spatial_key
