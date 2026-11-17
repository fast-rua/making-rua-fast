"""Modules imported on first use: `python -m robot wait / debug / pause` (robot/__main__.py) need only the standard
library, and numpy and scipy take ~0.6 s to import (0.7 s per call instead of 0.1 s)."""
import importlib
import sys


class Lazy:
    """Stands in for `module` (or `module:attr`) as the global `alias` of module `owner`. The first attribute read
    imports it and puts the real object in its place, so later reads cost nothing."""

    def __init__(self, target: str, owner: str, alias: str):
        self._target, self._owner, self._alias = target, owner, alias

    def __getattr__(self, name):
        mod, _, attr = self._target.partition(":")
        obj = importlib.import_module(mod)
        obj = getattr(obj, attr) if attr else obj
        setattr(sys.modules[self._owner], self._alias, obj)
        return getattr(obj, name)
