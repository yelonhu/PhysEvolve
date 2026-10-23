"""Question-family definitions for Stage-2 self-play.

Each active family maps to a scene template and supplies question-shape guidance
for the Proposer. The template verifies feasibility, derives answer semantics,
and constructs the final question and options. One simulated scene can yield
multiple questions; counterfactual and inverse families also use branch runs.

Frozen families remain registered but are excluded from active training. P1
entries reserve interfaces for inactive or incomplete families; their metadata
describes the available implementation and any required extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class QuestionFamily:
    """One PhysEvolve question shape bound to a verifying scene template.

    Attributes:
        name: Unique family key (the sampling vocabulary unit).
        skill_card: Informational label only;
            the verifier is the ``scene_kind`` scene template, not a card.
        tier: ``"P0"`` (a real verifier exists, family is trainable) or
            ``"P1"`` (reserved stub; needs renderer metadata, never active).
        priority: When True the family is one the curriculum should push
            hardest; the sampler applies a multiplicative boost.
        recommended_phrasings: Subset of the card's ``phrasings`` the
            Proposer should prefer for this family. Each entry must be a
            declared phrasing of ``skill_card``.
        stem_exemplars: Example question strings shown to the Proposer as
            Expected Grounding for this family's surface form.
        option_style: Human-readable description of the option layout (e.g.
            "3 named region labels + a 'Cannot be determined' escape").
        evidence_rule: Human-readable description of how the verifier grounds
            the answer (reinforces that the executor, not the Proposer,
            extracts ``correct_idx``).
        required_extension: For P1 families, the renderer / executor metadata
            that must ship before the family can become P0. Empty for P0.
        good_scene_pattern: Short description of the reusable, high-quality
            scene a good question of this family is built on (injected into
            the Proposer prompt). Empty when not authored.
        failure_modes: Common ways a Proposer botches this family (injected
            into the prompt as "avoid" bullets). Empty when not authored.
    """

    name: str
    skill_card: str
    tier: str
    priority: bool
    recommended_phrasings: tuple[str, ...]
    stem_exemplars: tuple[str, ...]
    option_style: str
    evidence_rule: str
    required_extension: str = ""
    good_scene_pattern: str = ""
    failure_modes: tuple[str, ...] = field(default_factory=tuple)
    # Every active family is served by the scene-template (multi-GT) path: the
    # Proposer authors one reusable scene + SCENE_MANIFEST of this ``scene_kind``
    # and the executor auto-derives the question. ``skill_card`` is now only an
    # informational label (the verifier is the scene template, not a card).
    scene_kind: Optional[str] = None
    # Frozen families remain registered but are excluded by
    # ``active_families`` (moved out of active training / golden export). Their
    # scene template is still registered so the logic is preserved.
    frozen: bool = False

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("QuestionFamily.name must be a non-empty string")
        if not self.skill_card or not isinstance(self.skill_card, str):
            raise ValueError(
                f"family {self.name!r}: skill_card must be a non-empty string"
            )
        if self.tier not in ("P0", "P1"):
            raise ValueError(
                f"family {self.name!r}: unknown tier {self.tier!r} "
                "(expected 'P0' or 'P1')"
            )
        if not self.stem_exemplars:
            raise ValueError(
                f"family {self.name!r}: stem_exemplars must be non-empty"
            )
        if self.tier == "P1" and self.priority:
            raise ValueError(
                f"family {self.name!r}: P1 families cannot be priority"
            )


# ─── Family registry ────────────────────────────────────────────────
#
# Active P0 families with priority=True receive the curriculum's priority
# multiplier. Other active families retain their configured sampling weights.

_THROWING_FINAL_DESTINATION = QuestionFamily(
    name="throwing_final_destination",
    skill_card="throwing_destination",
    tier="P0",
    priority=True,
    recommended_phrasings=("forward", "fate"),
    stem_exemplars=(
        "What is the final destination of the red ball?",
        "What will be the final position of the orange ball?",
        "Where will the white ball finally drop?",
        "What will happen to the green ball eventually?",
        "What will happen to the blue ball in the end?",
    ),
    option_style=(
        "Named regions: 'Left pit' / 'Right pit' / 'Middle pit' plus an "
        "'It remains uncertain' escape (slot 4)."
    ),
    evidence_rule=(
        "The executor reads the ball's final position from the trajectory and "
        "the GT is the unique pit (the visible GAP between platforms whose "
        "declared x-interval contains it); the ball must be at rest in that gap "
        "below the platform tops (final speed < 0.30 m/s, not ejected)."
    ),
    good_scene_pattern="",
    failure_modes=(
        "Declaring a pit that overlaps a platform or is not anchored to a "
        "platform edge - the pit must be a real visible gap.",
        "Writing SCENE_MANIFEST['pits'] before computing platform x-spans; "
        "each pit must be the open gap between sorted platform edges.",
        "Deciding the destination by raw launch speed instead of geometry.",
        "Ball still moving / ejected at the end - it never settles in a gap.",
        "Ball rests ON a platform / outside every pit interval.",
        "Colour-naming the ball body (e.g. green_ball) instead of a NEUTRAL "
        "name (ball_a) with a distinct rgba - the colour word in the question "
        "is derived from the rgba, and colour-baked names are rejected.",
    ),
    scene_kind="throwing_pit_board",
)

_THROWING_CURRENT_RUN_JUDGMENT = QuestionFamily(
    name="throwing_current_run_judgment",
    skill_card="throwing_run_judgment",
    tier="P0",
    priority=True,
    recommended_phrasings=("yes_no", "wrong_pit"),
    stem_exemplars=(
        "Will the white ball eventually drop into the right pit?",
        "Does the green ball end up in the left pit?",
        "Will the blue ball finally drop into the wrong pit?",
        "Will the orange ball finally drop into the left pit or another place?",
        "What will happen to the purple ball ultimately?",
    ),
    option_style="Yes / No / Maybe / Cannot be determined.",
    evidence_rule=(
        "Verifiable from the SINGLE actual run: GT = 'Yes' iff the ball comes "
        "to rest in the named target pit (the visible gap), else 'No' (the ball "
        "must have settled, not be ejected). This is NOT a hypothetical - it "
        "asks what truly happens in the video."
    ),
    good_scene_pattern="",
    failure_modes=(
        "Declaring a pit that is not a real gap between platforms.",
        "Skipping the platform x-span check before writing pit intervals; "
        "pits must be gaps, never platform footprints.",
        "Asking a hypothetical ('if we removed the wall...') - that is the "
        "counterfactual family, generated separately by the scene template.",
        "Ball not settled / ejected so the yes/no is ambiguous.",
        "target pit not one of the declared pit keys (left / right / middle).",
    ),
    scene_kind="throwing_pit_board",
)

_COLLISION_FIRST_CONTACT = QuestionFamily(
    name="collision_first_contact",
    skill_card="collision_first_contact",
    tier="P0",
    priority=False,
    recommended_phrasings=("forward", "counterfactual"),
    stem_exemplars=(
        "Which object hits the target first?",
        "If the red ball were removed, which object would hit the target "
        "first?",
    ),
    option_style="The four candidate body names (A/B/C/D = candidate order).",
    evidence_rule=(
        "The executor reads collision times from the trajectory; the GT is "
        "the candidate with the earliest first contact (distinct first-contact "
        "times must be >= 2 sim-steps apart)."
    ),
    good_scene_pattern=(
        "See collision_arena PLAYBOOK: four dynamic candidates, clear contact "
        "ordering, and geometry-driven timing."
    ),
    failure_modes=(
        "Making one candidate obviously fastest so the answer is trivial.",
        "Two candidates hitting within < 2 sim-steps of each other (ambiguous "
        "order).",
        "A candidate that never reaches the target.",
    ),
    scene_kind="collision_arena",
    frozen=True,
)

_COLLISION_FINAL_STATE = QuestionFamily(
    name="collision_final_state",
    skill_card="collision_final_state",
    tier="P0",
    priority=True,
    recommended_phrasings=("forward",),
    stem_exemplars=(
        "Which phrase best describes the final pose of the green pillar?",
        "What happens to the yellow pillar?",
        "What happens to the blue plate?",
        "After the collision involving the pillar, which phrase best "
        "describes its final state?",
    ),
    option_style=(
        "Four pose/state phrases: standing upright / still moving / lying "
        "horizontally / knocked out of the workspace."
    ),
    evidence_rule=(
        "The executor reads the body's final BodyStatus (stable / moving / "
        "fallen / ejected) from the trajectory; borderline speeds are "
        "rejected as ambiguous."
    ),
    good_scene_pattern=(
        "See collision_arena PLAYBOOK: trigger -> intermediate -> target, with "
        "one unambiguous final target state."
    ),
    failure_modes=(
        "Final state on a borderline (slowly drifting -> stable vs moving "
        "ambiguity).",
        "Using huge impact speed as the only reason for the outcome.",
        "Querying a body whose end state is visually unclear in the video.",
        "Target state unchanged because the trigger missed the chain.",
    ),
    scene_kind="collision_arena",
)

_MOTION_STATE_CHANGE = QuestionFamily(
    name="motion_state_change",
    skill_card="motion_state_change",
    tier="P0",
    priority=True,
    recommended_phrasings=("forward", "speed_profile", "pairwise_speed"),
    stem_exemplars=(
        "How does the box's motion change during the clip?",
        "Which best describes what happens to the ball's motion?",
        "What is the relationship between the speeds of the two objects "
        "after they begin to fall?",
        "What is the relationship between the speeds of the two balls in the "
        "video?",
        "How does the speed of the object change before it hits the ground?",
        "How does the speed of the ball change as it moves along the track?",
    ),
    option_style=(
        "Three supported surfaces: forward motion-change buckets; speed-profile "
        "options 'It accelerates' / 'It decelerates' / 'It first accelerates and "
        "then decelerates' / 'It remains constant'; pairwise-speed options "
        "'both same speed' / '{a} consistently faster' / '{b} consistently "
        "faster' / 'initially one faster, but the other eventually surpasses it'."
    ),
    evidence_rule=(
        "The executor reads initial, max, and final speeds from SimStateView: "
        "forward buckets use initial vs final speed; speed_profile uses initial "
        "/ max / final speed to distinguish accelerate, decelerate, peak-then-"
        "slow, or constant; pairwise_speed compares initial and final speed "
        "ordering for same / consistently faster / crossover-surpass. The "
        "motion_target must start with speed >= 0.10 m/s."
    ),
    good_scene_pattern=(
        "See motion_lab wiring: ramp, friction, rebound, or matched two-body "
        "lanes create a clear speed profile or pairwise speed relation."
    ),
    failure_modes=(
        "Target starts nearly at rest (initial speed < 0.10) so 'change' is "
        "undefined.",
        "Motion arc straddling two buckets (e.g. barely slows) - ambiguous.",
        "Making the answer obvious only because one body starts far faster.",
        "Uniform-speed coasting with no contact, friction, or gravity change - "
        "final bucket equals initial (weak question).",
        "Using only a friction_lane pair where both bodies keep constant speed "
        "throughout.",
    ),
    scene_kind="motion_lab",
)

# Non-priority P0 families — kept active so coverage stays broad.

_COLLISION_FINAL_RELATION = QuestionFamily(
    name="collision_final_relation",
    skill_card="collision_event_existence",
    tier="P0",
    priority=False,
    recommended_phrasings=(
        "forward",
        "encounter",
        "between",
        "binary_collide",
        "position",
    ),
    stem_exemplars=(
        "What is the final interaction between the blue plate and the brown "
        "pillar?",
        "What is the final position of the orange pillar in relation to the "
        "purple plate?",
        "What happens when the brown plate encounters the gray pillar?",
        "Does the white plate collide with the brown pillar?",
        "What happens between the blue plate and the pink pillar?",
        "Does the green pillar finally touch the yellow plate?",
        "Does the red plate eventually make contact with the blue pillar?",
    ),
    option_style=(
        "Pairwise relations: ends in contact / never touches / passes by / "
        "vertical separation."
    ),
    evidence_rule=(
        "The executor checks view.collided(a, b) and the trajectory min-"
        "distance / final z-gap; the Proposer's declared relation must match "
        "the simulated outcome."
    ),
    scene_kind="collision_arena",
)

_MOTION_ATTRIBUTION = QuestionFamily(
    name="motion_attribution",
    skill_card="motion_attribution",
    tier="P0",
    priority=False,
    recommended_phrasings=("attribution", "causal"),
    stem_exemplars=(
        "Which object experienced a change in motion as a result of the "
        "hammer strike?",
        "Whose motion is most clearly affected by the action of the striker?",
    ),
    option_style="The four candidate body names.",
    evidence_rule=(
        "The GT is argmax delta-speed over candidates that collided with the "
        "trigger; the top two must differ by >= 0.30 m/s."
    ),
    good_scene_pattern=(
        "See collision_arena PLAYBOOK: colour-named movers plus one dominant "
        "delta-speed candidate."
    ),
    failure_modes=(
        "Naming candidates non-colour (e.g. peg_a / peg_b) so the option text "
        "does not match the coloured bodies the Solver sees.",
        "Two candidates changing motion by a similar amount (attribution "
        "ambiguous - the top two must differ by >= 0.30 m/s).",
        "No candidate that stays still, so there is no real attribution choice.",
        "Trigger never contacts any mover - attribution is undefined.",
    ),
    scene_kind="collision_arena",
)

_MOTION_PHENOMENON_CLASSIFICATION = QuestionFamily(
    name="motion_phenomenon_classification",
    skill_card="motion_phenomenon_classification",
    tier="P0",
    priority=False,
    recommended_phrasings=("categorical", "reasoning"),
    stem_exemplars=(
        "Which phenomenon can be observed with the ball?",
        "Which best describes the movement of the cylindrical objects?",
    ),
    option_style=(
        "Four motion classes: linear / uniform circular / pendulum / "
        "oscillatory."
    ),
    evidence_rule=(
        "The Proposer declares expected_class; the executor verifies it is "
        "consistent with coarse trajectory signals from SimStateView."
    ),
    scene_kind="motion_phenomenon_classification",
    frozen=True,
)

_MASS_COMPARISON_BY_DYNAMICS = QuestionFamily(
    name="mass_comparison_by_dynamics",
    skill_card="mass_comparison_by_dynamics",
    tier="P0",
    priority=False,
    recommended_phrasings=("comparison", "judgment"),
    stem_exemplars=(
        "Which object is heavier, the red block or the blue block?",
    ),
    option_style="body_a / body_b / same / cannot be determined.",
    evidence_rule=(
        "The Proposer declares true_masses; the GT is the heavier body, "
        "cross-checked against the post-collision velocity ratio."
    ),
    scene_kind="collision_arena",
    frozen=True,
)

_FRICTION_OR_RESTITUTION_IDENTIFICATION = QuestionFamily(
    name="friction_or_restitution_identification",
    skill_card="friction_or_restitution_identification",
    tier="P0",
    priority=False,
    recommended_phrasings=("comparison", "perception"),
    stem_exemplars=(
        "Which surface has higher friction?",
        "Which ball has higher bounciness (restitution)?",
    ),
    option_style="body_a / body_b / same / cannot be determined.",
    evidence_rule=(
        "Two matched bodies differ only in the queried attribute; the GT is "
        "the extreme body, validated against travel distance / bounce height."
    ),
    scene_kind="motion_lab",
    frozen=True,
)

# ─── Throwing counterfactual / inverse (active branch-sim families) ───

_COUNTERFACTUAL_DESTINATION = QuestionFamily(
    name="counterfactual_destination",
    skill_card="counterfactual_destination",
    tier="P0",
    priority=False,
    recommended_phrasings=("counterfactual",),
    stem_exemplars=(
        "If we removed the blue floating wall and other balls, which pit "
        "would the green ball drop into?",
    ),
    option_style="Named regions (left / right / middle pit) + escape.",
    evidence_rule=(
        "Branch simulation: re-runs the scene with the named obstacle (and the "
        "other balls) removed and reads the counterfactual destination from the "
        "modified run. Now grounded by the throwing_pit_board scene template; "
        "the standalone single-question card path is not used."
    ),
    failure_modes=(
        "Declaring pit intervals that overlap platforms or sit off-screen; "
        "branch answers still use the same visible gap geometry.",
        "Declaring counterfactual_destination with fewer than two fixed walls, "
        "leaving no clear removable deflector.",
        "Putting weak decorative walls first in SCENE_MANIFEST['walls']; the "
        "first two entries are the shared counterfactual/inverse levers.",
        "Using colour-baked ball names instead of neutral ball_a / ball_b names "
        "with distinct rgba.",
    ),
    # Grounded via the scene-template multi-GT path (executor branch sim): the
    # verifier-grounded version lives in
    # physevolve/skills/scene_templates/throwing_pit_board.py.
    required_extension="",
    scene_kind="throwing_pit_board",
)

_INVERSE_INTERVENTION = QuestionFamily(
    name="inverse_intervention",
    skill_card="throwing_judgment",
    tier="P0",
    priority=False,
    recommended_phrasings=("judgment", "inverse"),
    stem_exemplars=(
        "What can we do to make the purple ball drop into the right pit?",
        "Which modification would make the orange ball settle in the left pit?",
    ),
    option_style=(
        "Intervention labels: remove obstacle A / B / both / leave unchanged."
    ),
    evidence_rule=(
        "Branch simulation: re-runs the scene under each candidate intervention "
        "(remove wall A / B / both, each also removing the other balls) and "
        "picks the unique intervention that lands the ball in the target region. "
        "Now grounded by the throwing_pit_board scene template -- the Proposer "
        "no longer self-declares intervention outcomes."
    ),
    failure_modes=(
        "Declaring pit intervals before checking platform x-spans; inverse "
        "targets must be real visible gaps.",
        "Providing fewer than two fixed walls, so remove-a / remove-b / both "
        "interventions cannot be grounded.",
        "Putting non-causal walls first in SCENE_MANIFEST['walls']; the first "
        "two entries define inverse levers A and B.",
        "Making every branch land in the same pit, leaving no unique "
        "intervention answer.",
    ),
    # Grounded via the scene-template multi-GT path (executor branch sim); the
    # legacy throwing_judgment self-eval card is NOT used in the manifest path.
    required_extension="",
    scene_kind="throwing_pit_board",
)

# ─── P1 families (reserved; need renderer/executor metadata) ──────────
#
# These are NEVER returned by active_families(); resolving one in the active
# training path raises (mirrors the P1 skill-card NotImplementedError contract).
# Each records the extension required to graduate it to P0. See the
# Stage-2 manifest-unification plan F for the camera_depth /
# viewpoint landing design.

# camera_depth has a geometric executor handler (GT = argmin/argmax distance
# from the rendered camera eye to each candidate body);
# camera/view families are excluded from active training.
_CAMERA_DEPTH_CLOSEST_OBJECT = QuestionFamily(
    name="camera_depth_closest_object",
    skill_card="camera_depth_closest_object",
    tier="P1",
    priority=False,
    recommended_phrasings=("static",),
    stem_exemplars=(
        "Which object is the camera closest to?",
        "Determine which point is nearest to the camera.",
        "What is farthest from the key in the picture?",
    ),
    option_style="The four candidate body names (closest, or farthest).",
    evidence_rule=(
        "The executor reads the rendered camera's eye position and each "
        "candidate's position from the scene geometry; the GT is the body with "
        "the min (closest) or max (farthest) Euclidean distance. The two "
        "extreme candidates must differ by > 0.15 m."
    ),
    good_scene_pattern=(
        "Four clearly-coloured, well-separated bodies at different distances "
        "from one camera (api.add_camera); the question asks which is closest "
        "(or farthest). The scene is static - depth is a spatial relationship."
    ),
    failure_modes=(
        "Two candidates almost equidistant from the camera (ambiguous).",
        "Naming candidates non-colour so the option text mismatches the video.",
        "No camera added (the default camera may not frame the bodies well).",
    ),
    required_extension=(
        "geometric camera_depth executor exists, but camera/viewpoint families "
        "are excluded from active training"
    ),
    scene_kind="camera_depth_closest_object",
    frozen=True,
)

_SCREEN_LEFT_RIGHT_LOCATION = QuestionFamily(
    name="screen_left_right_location",
    skill_card="visual_left_right_top_bottom",
    tier="P1",
    priority=False,
    recommended_phrasings=("static",),
    stem_exemplars=(
        "Which point corresponds to the reference point P?",
        "How does the moon pass through the sun (from which corner to which "
        "corner)?",
    ),
    option_style="Screen-relative labels (left / right / top / bottom; corners).",
    evidence_rule=(
        "Would need a projected 2D bbox per body to know screen-relative "
        "position and on-screen motion direction."
    ),
    # TODO(P1): requires a projected 2D bbox API on SimStateView.
    required_extension="projected 2D bbox API on SimStateView",
)

_VIEWPOINT_MOTION_BASIC = QuestionFamily(
    name="viewpoint_motion_basic",
    skill_card="viewpoint_motion_basic",
    tier="P1",
    priority=False,
    recommended_phrasings=("reasoning",),
    stem_exemplars=(
        "Which camera change can produce the effect in the video?",
        "Which camera modification could result in this outcome?",
    ),
    option_style=(
        "Camera-motion descriptions: rotate about vertical/horizontal axis, "
        "move parallel to a line, move closer/farther, move up/down."
    ),
    evidence_rule=(
        "Would need a camera-trajectory timeline to know how the camera moved."
    ),
    # TODO(P1): requires api.set_camera_trajectory(...) + renderer camera-pose
    # timeline + SimStateView.camera_trajectory().
    required_extension=(
        "api.set_camera_trajectory(...) + renderer camera-pose timeline + "
        "SimStateView.camera_trajectory()"
    ),
)

_IMAGE_OPTION_TEMPORAL_ORDER = QuestionFamily(
    name="image_option_temporal_order",
    skill_card="image_option_temporal_order",
    tier="P1",
    priority=False,
    recommended_phrasings=("image_options",),
    stem_exemplars=(
        "Following the content of the video, which option's corresponding "
        "picture will happen first?",
    ),
    option_style="Four image options (frames at distinct sim-step indices).",
    evidence_rule=(
        "Would need multi-frame dump + multi-image option rendering in the "
        "Solver prompt."
    ),
    # TODO(P1): requires executor multi-frame dump + vqa.build_observation
    # 4-image bundle + a Solver VQA_TEMPLATE image-option variant.
    required_extension=(
        "executor multi-frame dump + 4-image option bundle + Solver "
        "image-option template"
    ),
)


_FAMILY_LIST: tuple[QuestionFamily, ...] = (
    # Priority P0 (pushed hardest).
    _THROWING_FINAL_DESTINATION,
    _THROWING_CURRENT_RUN_JUDGMENT,
    _COLLISION_FIRST_CONTACT,
    _COLLISION_FINAL_STATE,
    _MOTION_STATE_CHANGE,
    # Non-priority P0 (coverage).
    _COLLISION_FINAL_RELATION,
    _MOTION_ATTRIBUTION,
    _MOTION_PHENOMENON_CLASSIFICATION,
    _MASS_COMPARISON_BY_DYNAMICS,
    _FRICTION_OR_RESTITUTION_IDENTIFICATION,
    # Active branch-sim P0 families (non-priority coverage).
    _COUNTERFACTUAL_DESTINATION,
    _INVERSE_INTERVENTION,
    # P1 reserved (never active).
    _CAMERA_DEPTH_CLOSEST_OBJECT,
    _SCREEN_LEFT_RIGHT_LOCATION,
    _VIEWPOINT_MOTION_BASIC,
    _IMAGE_OPTION_TEMPORAL_ORDER,
)

QUESTION_FAMILY_REGISTRY: dict[str, QuestionFamily] = {}
for _family in _FAMILY_LIST:
    if _family.name in QUESTION_FAMILY_REGISTRY:
        raise ValueError(f"duplicate question family name {_family.name!r}")
    QUESTION_FAMILY_REGISTRY[_family.name] = _family
del _family

# Branch-elision families: counterfactual / inverse items grounded by extra
# simulator rollouts. The w/o-branch-elision ablation drops this pair from
# sampling, executor branch sims, and Solver payloads together.
BRANCH_ELISION_FAMILIES: frozenset[str] = frozenset(
    {
        "counterfactual_destination",
        "inverse_intervention",
    }
)


# ─── Public API ───────────────────────────────────────────────────────


def lookup_family(name: str) -> QuestionFamily:
    """Return the :class:`QuestionFamily` named ``name`` or raise ValueError."""

    try:
        return QUESTION_FAMILY_REGISTRY[name]
    except KeyError:
        available = sorted(QUESTION_FAMILY_REGISTRY)
        raise ValueError(
            f"unknown question family {name!r}; available: {available}"
        ) from None


def active_families(*, include_p1: bool = False) -> tuple[str, ...]:
    """Return the sorted active family vocabulary.

    Frozen families are always excluded (moved out of active training). P1
    families are excluded unless ``include_p1`` is True (only useful for
    forward-compatibility audits / tests; never for live training).
    """

    return tuple(
        sorted(
            name
            for name, fam in QUESTION_FAMILY_REGISTRY.items()
            if not fam.frozen and (include_p1 or fam.tier == "P0")
        )
    )


def priority_families() -> tuple[str, ...]:
    """Return the sorted set of priority P0 family names."""

    return tuple(
        sorted(
            name
            for name, fam in QUESTION_FAMILY_REGISTRY.items()
            if fam.priority
        )
    )


def validate_registry() -> None:
    """Assert every family is consistently wired to its scene-template verifier.

    After the manifest unification there is a single verifier kind: a
    :class:`~physevolve.skills.scene_templates.SceneTemplate`.

    * Families with a ``scene_kind`` (every active family) must point at a
      registered scene template that actually serves them.
    * Families without a ``scene_kind`` must be reserved P1 stubs (no template
      yet); a P0 family without a template is a wiring bug.

    Run by tests / the step builder; not at import time (template registry
    imports are deferred).

    Raises:
        ValueError: On any inconsistency.
    """

    from physevolve.skills.scene_templates import get_scene_template_registry

    scene_reg = get_scene_template_registry()
    for name, family in QUESTION_FAMILY_REGISTRY.items():
        if family.scene_kind:
            template = scene_reg.lookup(family.scene_kind)  # raises if unknown
            if family.frozen:
                continue
            if name not in template.families:
                raise ValueError(
                    f"family {name!r} declares scene_kind "
                    f"{family.scene_kind!r} which does not serve it "
                    f"(template serves {template.families})"
                )
            continue
        if family.tier != "P1":
            raise ValueError(
                f"family {name!r} is {family.tier!r} but has no scene_kind; "
                "only reserved P1 stubs may lack a scene template"
            )


__all__ = [
    "BRANCH_ELISION_FAMILIES",
    "QUESTION_FAMILY_REGISTRY",
    "QuestionFamily",
    "active_families",
    "lookup_family",
    "priority_families",
    "validate_registry",
]
