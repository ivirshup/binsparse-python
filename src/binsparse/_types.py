from __future__ import annotations

try:
    from zarr.core import Array as ZarrArray
    from zarr.hierarchy import Group as ZarrGroup

except ImportError:

    class ZarrArray:
        @staticmethod
        def __repr__():
            return "mock zarr.core.Array"

    class ZarrGroup:
        @staticmethod
        def __repr__():
            return "mock zarr.core.Group"


ZarrTypes = ZarrArray | ZarrGroup

try:
    from h5py import Dataset as H5Array
    from h5py import Group as H5Group

except ImportError:

    class H5Array:
        @staticmethod
        def __repr__():
            return "mock h5py.Dataset"

    class H5Group:
        @staticmethod
        def __repr__():
            return "mock h5py.Group"


H5Types = H5Array | H5Group

ArrayTypes = ZarrArray | H5Array
GroupTypes = ZarrGroup | H5Group
