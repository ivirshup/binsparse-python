import h5py
import pytest
import zarr
from scipy import sparse

import binsparse
from binsparse._testing import assert_equal


def test_package_has_version():
    assert binsparse.__version__ is not None


@pytest.fixture(params=["h5", "zarr"])
def store(request, tmp_path):
    store_str = request.param
    if store_str == "h5":
        filename = "tmp.h5"
        return h5py.File(tmp_path / filename, "w")["/"]
    else:
        return zarr.group()


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
def test_basic_roundtrip(store, format):
    orig = sparse.random(100, 100, density=0.1, format=format)
    binsparse.write(store, "X", orig)
    from_disk = binsparse.read(store["X"])

    assert_equal(orig, from_disk)
