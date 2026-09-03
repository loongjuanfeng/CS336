"""Compatibility facade for the official CS336 adapter contract.

The copied official suite lives in :mod:`tests.official` so it can coexist
with this repository's original tests.  Keeping this module at the canonical
upstream path also lets the official tests be run or overlaid directly under
``tests/`` without requiring another adapter implementation.
"""

from .official.adapters import *
