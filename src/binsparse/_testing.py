from functools import singledispatch

import numpy as np
from scipy import sparse


def assert_equal(a: sparse.spmatrix, b):
    assert type(a) == type(b), f"types differ: {type(a)} != {type(b)}"
    assert a.shape == b.shape, f"shapes differ: {a.shape} != {b.shape}"
    _assert_equal(a, b)


@singledispatch
def _assert_equal(a, b):
    raise NotImplementedError(f"no implementation for type {type(a)}")


@_assert_equal.register(sparse.csr_matrix)
@_assert_equal.register(sparse.csc_matrix)
def _(a, b):
    for attr in ["indptr", "indices", "data"]:
        np.testing.assert_equal(getattr(a, attr), getattr(b, attr))


@_assert_equal.register(sparse.coo_matrix)
def _(a, b):
    for attr in ["row", "col", "data"]:
        np.testing.assert_equal(getattr(a, attr), getattr(b, attr))
