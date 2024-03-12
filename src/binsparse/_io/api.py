from collections.abc import Mapping
from types import MappingProxyType

import graphblas as gb
from scipy import sparse

from binsparse._io.methods import (
    read_coo,
    read_csc,
    read_csr,
    write_coo,
    write_csc,
    write_csr,
)
from binsparse._io.utils import read_attr
from binsparse._types import GroupTypes


def read(group: GroupTypes, struct: str | None = "scipy") -> gb.Matrix | sparse.spmatrix:
    """Read a sparse matrix from a store.

    Parameters
    ----------
    group
        A Zarr or h5py group to read the matrix from.
    """
    metadata = read_attr(group, "binsparse")
    if metadata["format"] == "CSR":
        return read_csr(group, struct=struct)
    elif metadata["format"] == "CSC":
        return read_csc(group, struct=struct)
    elif metadata["format"] == "COO":
        return read_coo(group, struct=struct)
    else:
        raise NotImplementedError(f"no implementation for format {metadata['format']}")


def write(store: GroupTypes, key: str, x: sparse.spmatrix, *, dataset_kwargs: Mapping = MappingProxyType({})):
    """Write a sparse matrix to a store.

    Parameters
    ----------
    store
        A Zarr or h5py group.
    key
        The key to write the matrix to.
    x
        The matrix to write.
    dataset_kwargs
        Keyword arguments to pass to the dataset creation function.
    """
    if isinstance(x, sparse.csr_matrix):
        write_csr(store, key, x, dataset_kwargs=dataset_kwargs)
    elif isinstance(x, sparse.csc_matrix):
        write_csc(store, key, x, dataset_kwargs=dataset_kwargs)
    elif isinstance(x, sparse.coo_matrix):
        write_coo(store, key, x, dataset_kwargs=dataset_kwargs)
    else:
        raise NotImplementedError(f"no implementation for type {type(x)}")
