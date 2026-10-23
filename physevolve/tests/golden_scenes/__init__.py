"""Ten hand-written PhysDSL programs.

Each subpackage (``01_...`` through ``10_...``) ships a triple:

* ``scene.py`` — Proposer-style source with ``build_scene(api)``.
* ``expected.json`` — per-body final grid, status, and collision count.
* ``test_{name}.py`` — runs the scene through ``run_executor`` and
  asserts the expected outcome within grid-distance tolerance.

The conftest at this level exposes shared helpers (``run_golden``,
``load_expected``, ``assert_body_grid_within``) so each per-scene
test file is a thin wrapper.
"""
