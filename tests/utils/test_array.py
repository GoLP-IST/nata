# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nata.utils.array import expand_ellipsis


@pytest.mark.parametrize(
    "indexing, dimension, expanded_indexing",
    [
        (np.s_[0], 3, np.index_exp[0]),
        (np.s_[:], 5, np.index_exp[:]),
        (np.s_[...], 1, np.index_exp[:]),
        (np.s_[0, ...], 4, np.index_exp[0, :, :, :]),
        (np.s_[np.newaxis, ...], 3, np.index_exp[np.newaxis, :, :, :]),
        (np.s_[..., 0], 4, np.index_exp[:, :, :, 0]),
        (np.s_[0, ..., 0], 3, np.index_exp[0, :, 0]),
        (np.s_[np.newaxis, ..., 0], 3, np.index_exp[np.newaxis, :, :, 0]),
        (np.s_[0, ..., np.newaxis], 3, np.index_exp[0, :, :, np.newaxis]),
        (
            np.s_[np.newaxis, ..., np.newaxis],
            3,
            np.index_exp[np.newaxis, :, :, :, np.newaxis],
        ),
        (
            np.index_exp[np.newaxis, ..., 0],
            3,
            np.index_exp[np.newaxis, :, :, 0],
        ),
        (np.s_[np.newaxis, ...], 0, np.index_exp[np.newaxis]),
        (np.s_[np.newaxis, ...], 1, np.index_exp[np.newaxis, :]),
    ],
    ids=[
        "3d & [int]",
        "5d & [:]",
        "1d & [...]",
        "4d & [int, ...]",
        "3d & [None, ...]",
        "4d & [..., int]",
        "3d & [int, ..., int]",
        "3d & [None, ..., int]",
        "3d & [int, ..., None]",
        "3d & [None, ..., None]",
        "3d & [(None, ..., 0)]",
        "0d & [None, ...]",
        "1d & [None, ...]",
    ],
)
def test_expand_ellipsis(indexing, dimension, expanded_indexing):
    assert expand_ellipsis(indexing, dimension) == expanded_indexing


def test_expand_ellipsis_raise():
    with pytest.raises(KeyError, match="one Ellipse '...' is allowed!"):
        expand_ellipsis((..., ...), 3)
