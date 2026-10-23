"""Shared zero-shot VLM evaluation utilities (packing, sharding, family generate).

Import submodules explicitly, e.g. ``evaluation.common.family_generate``.
This package ``__init__`` does not re-export ``common`` helpers, so PhysBench's
official adapter can import ``family_generate`` without a circular import.
"""

__all__: list[str] = []
