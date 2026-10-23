"""Deterministic, no-GPU mock backend for Stage-2 self-play probes.

The Stage-2 training entry uses this mock for ``--dry-run`` so Proposer /
Solver generations can be produced without loading the 8B model.

Routing is by ``GenerationSpec.rollout_metadata['role']``:

* ``role == 'proposer'`` — return one scene template golden variant (a runnable
  ``build_scene`` + literal ``SCENE_MANIFEST``) for the requested
  ``scene_kind`` / ``fewshot_index``, so the mock exercises the *whole* unified
  executor path (sim -> view -> scene-template verifier -> payloads).
* ``role == 'solver'`` — return the next letter in a cycling A/B/C/D queue so
  reward variance is non-zero across the batch without a real LLM.
"""

from __future__ import annotations

import textwrap
from collections import Counter
from dataclasses import dataclass, field

from physevolve.inference.hf_backend import GenerationResult


# Fallback Proposer rollout used when the requested ``scene_kind`` cannot be
# resolved to a template fewshot (missing template / missing fewshot). It
# degrades gracefully into a valid-but-minimal SCENE_MANIFEST scene so the probe
# surfaces a real executor result rather than crashing the mock.
_FALLBACK_PROPOSER_TEXT = textwrap.dedent(
    """\
    <propose>
    def build_scene(api):
        api.add_box("blue_box", pos=(-0.7, 0.0, 0.10), size=(0.10, 0.10, 0.10),
                    mass=1.0, rgba=(0.2, 0.4, 0.9, 1.0), friction=2.0,
                    restitution=0.0, fixed=False)
        api.set_velocity("blue_box", linear=(1.8, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
        api.set_duration(2.5)

    SCENE_MANIFEST = {
        "scene_kind": "motion_lab",
        "motion_target": "blue_box",
        "families": ["motion_state_change"],
    }
    </propose>
    """
).strip()


_SOLVER_LETTERS = ("A", "B", "C", "D")


def build_proposer_text_for_scene_kind(scene_kind: str, *, fewshot_index: int = 0) -> str:
    """Return one scene-template fewshot variant verbatim."""

    from physevolve.skills.scene_templates import (
        get_scene_template_registry,
        select_fewshot_example,
    )

    try:
        template = get_scene_template_registry().lookup(scene_kind)
    except ValueError:
        return _FALLBACK_PROPOSER_TEXT
    fewshot = select_fewshot_example(template, fewshot_index).strip()
    if "<propose>" not in fewshot:
        return _FALLBACK_PROPOSER_TEXT
    return fewshot


@dataclass
class Stage2MockBackend:
    """Scene-template-driven deterministic mock backend (no GPU, no model load).

    Args:
        n_solver: Ignored (kept for back-compat). Solver answers are produced
            on demand by cycling A/B/C/D, so the multi-GT path (many questions
            per scene) cannot exhaust a pre-sized queue.
    """

    n_solver: int = 0
    _proposer_serve_count: "Counter[str]" = field(default_factory=Counter)
    _solver_serve_count: int = 0

    def __enter__(self) -> "Stage2MockBackend":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def generate_batch(self, specs):  # type: ignore[no-untyped-def]
        results: list[GenerationResult] = []
        for spec in specs:
            metadata = spec.rollout_metadata or {}
            role = metadata.get("role", "proposer")
            if role == "proposer":
                scene_kind = metadata.get("scene_kind")
                if scene_kind:
                    text = build_proposer_text_for_scene_kind(
                        scene_kind,
                        fewshot_index=int(metadata.get("fewshot_index", 0)),
                    )
                    self._proposer_serve_count[f"scene:{scene_kind}"] += 1
                else:
                    text = _FALLBACK_PROPOSER_TEXT
                    self._proposer_serve_count["<fallback>"] += 1
            else:
                # Cycle A/B/C/D on demand so reward variance is non-zero and
                # the queue can never run out (multi-GT scenes ask many Qs).
                text = f"<answer>{_SOLVER_LETTERS[self._solver_serve_count % 4]}</answer>"
                self._solver_serve_count += 1
            results.append(
                GenerationResult(
                    text=text,
                    metadata={
                        "generated_token_count": 5,
                        "video_present": spec.video is not None,
                    },
                    generated_text=text,
                    prompt_text=spec.prompt,
                    prompt_tail=spec.prompt[-32:],
                )
            )
        return results


__all__ = [
    "Stage2MockBackend",
    "build_proposer_text_for_scene_kind",
]
