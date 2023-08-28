from __future__ import annotations

from functools import singledispatch

from binsparse._types import ArrayTypes, GroupTypes, H5Array, H5Group, ZarrArray, ZarrGroup


@singledispatch
def write_attrs(element: ArrayTypes | GroupTypes, key: str, value):
    raise NotImplementedError(f"no implementation for type {type(element)}")


@singledispatch
def read_attr(element: ArrayTypes | GroupTypes, key: str):
    raise NotImplementedError(f"no implementation for type {type(element)}")


@write_attrs.register(ZarrArray)
@write_attrs.register(ZarrGroup)
def _(element, key, value):
    element.attrs[key] = value


@read_attr.register(ZarrArray)
@read_attr.register(ZarrGroup)
def _(element, key) -> dict:
    return dict(element.attrs[key])


# def _write_attrs_to_h5(attrs, key, value):
#     if isinstance(value, dict):
#         attrs[key] = {}
#         for k, v in value.items():
#             _write_attrs_to_h5(attrs[key], k, v)
#     else:
#         attrs[key] = value


@write_attrs.register(H5Array)
@write_attrs.register(H5Group)
def _(element, key, value):
    import json

    element.attrs[key] = json.dumps(value)
    # _write_attrs_to_h5(element.attrs, key, value)


@read_attr.register(H5Array)
@read_attr.register(H5Group)
def _(element, key) -> dict:
    import json

    return json.loads(element.attrs[key])
