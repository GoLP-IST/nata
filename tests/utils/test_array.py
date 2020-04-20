# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nata.utils.array import expand_ellipsis


@pytest.mark.parametrize(
    "indexing, dimension, expanded_indexing",
    [
        (np.s_[0], 3, np.index_exp[0]),
        (np.s_[:], 10, np.index_exp[:]),
        (np.s_[0, ...], 4, np.index_exp[0, :, :, :]),
        (np.s_[np.newaxis, ...], 3, np.index_exp[np.newaxis, :, :]),
        (np.s_[..., 0], 4, np.index_exp[:, :, :, 0]),
        (np.s_[0, ..., 0], 3, np.index_exp[0, :, 0]),
        (np.s_[np.newaxis, ..., 0], 3, np.index_exp[np.newaxis, :, 0]),
        (np.s_[0, ..., np.newaxis], 3, np.index_exp[0, :, np.newaxis]),
        (
            np.s_[np.newaxis, ..., np.newaxis],
            3,
            np.index_exp[np.newaxis, :, np.newaxis],
        ),
        (np.index_exp[np.newaxis, ..., 0], 3, np.index_exp[np.newaxis, :, 0],),
    ],
    ids=[
        "int",
        ":",
        "int, ...",
        "None, ...",
        "..., int",
        "int, ..., int",
        "None, ..., int",
        "int, ..., None",
        "None, ..., None",
        "(None, ..., 0)",
    ],
)
def test_expand_ellipsis(indexing, dimension, expanded_indexing):
    assert len(expand_ellipsis(indexing, dimension)) == len(expanded_indexing)
    assert expand_ellipsis(indexing, dimension) == expanded_indexing
