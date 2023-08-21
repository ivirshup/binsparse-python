from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from scipy import sparse

from binsparse._io.utils import read_attr, write_attrs
from binsparse._types import GroupTypes

_DTYPE_STR_REGISTRY = {
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("bool"): "bint8",
}


def dtype_str(x: np.dtype) -> str:
    result = _DTYPE_STR_REGISTRY.get(x, None)
    if result is None:
        raise ValueError(f"unrecognized dtype: {x}")
    return result


def write_csr(store: GroupTypes, key: str, x: sparse.csr_matrix, *, dataset_kwargs: Mapping = MappingProxyType({})):
    """Write a CSR matrix to a store.

    Parameters
    ----------
    store
        A Zarr or h5py group.
    key
        The key to write the matrix to.
    x
        The matrix to write.
    """
    metadata = {
        "format": "CSR",
        "shape": list(x.shape),
        "data_types": {
            "pointers_to_1": dtype_str(x.indptr.dtype),
            "indices_1": dtype_str(x.indices.dtype),
            "values": dtype_str(x.data.dtype),
        },
    }

    group = store.require_group(key)
    write_attrs(group, "binsparse", metadata)

    store.create_dataset(f"{key}/pointers_to_1", data=x.indptr, **dataset_kwargs)
    store.create_dataset(f"{key}/indices_1", data=x.indices, **dataset_kwargs)
    store.create_dataset(f"{key}/values", data=x.data, **dataset_kwargs)


def read_csr(group: GroupTypes) -> sparse.csr_matrix:
    """Read a CSR matrix from a store.

    Parameters
    ----------
    group
        A Zarr or h5py group.

    Returns
    -------
    x
        The matrix.
    """
    metadata = read_attr(group, "binsparse")
    assert metadata["format"] == "CSR"
    shape = tuple(metadata["shape"])

    return sparse.csr_matrix(
        (
            group["values"][()],
            group["indices_1"][()],
            group["pointers_to_1"][()],
        ),
        shape=shape,
    )


def write_csc(store: GroupTypes, key: str, x: sparse.csc_matrix, *, dataset_kwargs: Mapping = MappingProxyType({})):
    """Write a CSR matrix to a store.

    Parameters
    ----------
    store
        A Zarr or h5py group.
    key
        The key to write the matrix to.
    x
        The matrix to write.
    """
    metadata = {
        "format": "CSC",
        "shape": list(x.shape),
        "data_types": {
            "pointers_to_1": dtype_str(x.indptr.dtype),
            "indices_1": dtype_str(x.indices.dtype),
            "values": dtype_str(x.data.dtype),
        },
    }

    group = store.require_group(key)
    write_attrs(group, "binsparse", metadata)

    store.create_dataset(f"{key}/pointers_to_1", data=x.indptr, **dataset_kwargs)
    store.create_dataset(f"{key}/indices_1", data=x.indices, **dataset_kwargs)
    store.create_dataset(f"{key}/values", data=x.data, **dataset_kwargs)


def read_csc(group: GroupTypes) -> sparse.csc_matrix:
    """Read a CSC matrix from a store.

    Parameters
    ----------
    group
        A Zarr or h5py group.

    Returns
    -------
    x
        The matrix.
    """
    metadata = read_attr(group, "binsparse")
    assert metadata["format"] == "CSC"
    shape = tuple(metadata["shape"])

    return sparse.csc_matrix(
        (
            group["values"][()],
            group["indices_1"][()],
            group["pointers_to_1"][()],
        ),
        shape=shape,
    )


def write_coo(store: GroupTypes, key: str, x: sparse.csc_matrix, *, dataset_kwargs: Mapping = MappingProxyType({})):
    """Write a CSR matrix to a store.

    Parameters
    ----------
    store
        A Zarr or h5py group.
    key
        The key to write the matrix to.
    x
        The matrix to write.
    """
    metadata = {
        "format": "COO",
        "shape": list(x.shape),
        "data_types": {
            "indices_0": dtype_str(x.row.dtype),
            "indices_1": dtype_str(x.col.dtype),
            "values": dtype_str(x.data.dtype),
        },
    }

    group = store.require_group(key)
    write_attrs(group, "binsparse", metadata)

    store.create_dataset(f"{key}/indices_0", data=x.row, **dataset_kwargs)
    store.create_dataset(f"{key}/indices_1", data=x.col, **dataset_kwargs)
    store.create_dataset(f"{key}/values", data=x.data, **dataset_kwargs)


def read_coo(group: GroupTypes):
    """Read a COO matrix from a store.

    Parameters
    ----------
    group
        A Zarr or h5py group.

    Returns
    -------
    x
        The matrix.
    """
    metadata = read_attr(group, "binsparse")
    assert metadata["format"] == "COO"
    shape = tuple(metadata["shape"])

    return sparse.coo_matrix(
        (
            group["values"][()],
            (
                group["indices_0"][()],
                group["indices_1"][()],
            ),
        ),
        shape=shape,
    )
