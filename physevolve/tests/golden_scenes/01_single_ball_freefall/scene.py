"""Golden scene 01 — single ball free-fall (sanity).

A ball at z=2 m falls onto the ground plane under default gravity.
After 1 s (well past touchdown for a 2 m drop), the ball should have
settled near z=radius.
"""


def build_scene(api):
    api.add_sphere("ball", pos=(0.0, 0.0, 2.0), radius=0.1, mass=1.0)
    api.set_duration(1.0)
