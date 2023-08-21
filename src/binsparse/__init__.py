from importlib.metadata import version

from binsparse._io.api import read, write

__version__ = version("binsparse")
del version
