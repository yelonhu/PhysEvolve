"""Top-level ``evaluation`` package.

This lightweight ``__init__.py`` lets bench subpackages (``evaluation.physbench``,
``evaluation.contphy``, ``evaluation.common``, ...) be imported via dotted
paths from launchers and tests.

No re-exports here — sub-packages are imported explicitly by consumers.
"""

__all__: list[str] = []
