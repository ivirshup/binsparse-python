import graphblas as gb
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


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_basic_roundtrip(store, fmt):
    orig = sparse.random(100, 100, density=0.1, format=fmt)
    binsparse.write(store, "X", orig)
    from_disk = binsparse.read(store["X"])

    assert_equal(orig, from_disk)


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_metadata(store, fmt):
    from binsparse._io.utils import read_attr

    orig = sparse.random(100, 100, density=0.1, format=fmt)
    binsparse.write(store, "X", orig)
    metadata = read_attr(store["X"], "binsparse")

    assert metadata["format"] == fmt.upper()
    assert metadata["shape"] == list(orig.shape)

    from binsparse._io.methods import _DTYPE_STR_REGISTRY

    for k, v in metadata["data_types"].items():
        assert v in _DTYPE_STR_REGISTRY.values(), f"unrecognized dtype for '{k}': {v}"


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_graphblas_structure(store, fmt):
    struct = "graphblas"
    orig = sparse.random(100, 100, density=0.1, format=fmt)
    binsparse.write(store, "X", orig)
    orig = gb.io.from_scipy_sparse(orig)
    from_disk = binsparse.read(store["X"], struct=struct)
    assert_equal(orig, from_disk)
