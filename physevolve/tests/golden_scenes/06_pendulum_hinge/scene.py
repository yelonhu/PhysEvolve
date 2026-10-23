"""Hinge pendulum with a fixed parent and a swinging child.

The scene checks parent/child MJCF nesting. Final grid comparisons allow one
cell of tolerance for differences in the pendulum phase.
"""


def build_scene(api):
    api.add_box(
        "anchor",
        pos=(0.0, 0.0, 2.0),
        size=(0.05, 0.05, 0.05),
        mass=1.0,
        fixed=True,
    )
    api.add_sphere(
        "bob",
        pos=(0.5, 0.0, 2.0),
        radius=0.05,
        mass=0.5,
    )
    api.add_hinge(
        name="pivot",
        parent="anchor",
        child="bob",
        pos=(0.0, 0.0, 2.0),
        axis=(0.0, 1.0, 0.0),
    )
    api.set_duration(2.0)
