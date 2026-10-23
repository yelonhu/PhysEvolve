"""Engine layer — pure physics, no RL, no VLM.

Environment variable setup runs at import time and MUST happen before any
``import mujoco`` anywhere in the process (see docs/architecture.md).
Using ``setdefault`` keeps explicit overrides by the
caller (e.g. ``MUJOCO_GL=osmesa`` on systems without EGL) functional.
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")
if os.environ["MUJOCO_GL"] in {"egl", "osmesa"}:
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MUJOCO_LOG_LEVEL", "ERROR")
