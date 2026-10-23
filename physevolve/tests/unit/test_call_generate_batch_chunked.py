"""Unit tests for ``call_generate_batch_chunked``.

The Stage-2 builder dispatches the (video-bearing) VQA solver rollout in
fixed-size chunks to bound the per-row multimodal preprocessing peak. The
critical invariant is that the rollout DataProto of *every* chunk is collected
immediately after that chunk runs -- a naive loop that collected once at the
end would let each chunk overwrite the backend's ``_last_rollout`` and silently
drop all but the final chunk from the PPO update. These tests pin both the
result ordering and the per-chunk collection behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from physevolve.inference.hf_backend import GenerationResult
from physevolve.training.online_step_builder import (
    GenerationSpec,
    call_generate_batch_chunked,
)


@dataclass
class _CountingBackend:
    """Fake backend that emits one counting rollout object per batch call.

    Each ``generate_batch`` call stores the prompts it served as the current
    ``_last_rollout``; ``consume_last_rollout`` drains it (mirroring the real
    trainer backend contract). This lets a test assert that no chunk's rollout
    was overwritten before being collected.
    """

    batch_sizes: list[int] = field(default_factory=list)
    _last_rollout: Any = None

    def generate_batch(self, specs: list[GenerationSpec]) -> list[GenerationResult]:
        spec_list = list(specs)
        self.batch_sizes.append(len(spec_list))
        # The "rollout" is just the list of prompts served in this chunk.
        self._last_rollout = [spec.prompt for spec in spec_list]
        return [
            GenerationResult(text=spec.prompt, metadata={"generated_token_count": 1})
            for spec in spec_list
        ]

    def consume_last_rollout(self) -> Any:
        rollout = self._last_rollout
        self._last_rollout = None
        return rollout


def _make_specs(n: int) -> list[GenerationSpec]:
    return [GenerationSpec(prompt=f"p{i:03d}") for i in range(n)]


def test_chunking_preserves_row_count_and_order() -> None:
    backend = _CountingBackend()
    specs = _make_specs(10)
    rollout_batches: list[object] = []

    results = call_generate_batch_chunked(backend, specs, rollout_batches, 4)

    assert [r.text for r in results] == [s.prompt for s in specs]
    assert backend.batch_sizes == [4, 4, 2]


def test_every_chunk_rollout_is_collected() -> None:
    backend = _CountingBackend()
    specs = _make_specs(10)
    rollout_batches: list[object] = []

    call_generate_batch_chunked(backend, specs, rollout_batches, 4)

    # One collected rollout per chunk (ceil(10 / 4) == 3), and together they
    # cover every prompt in dispatch order -- i.e. no chunk was overwritten.
    assert len(rollout_batches) == 3
    flattened = [prompt for chunk in rollout_batches for prompt in chunk]
    assert flattened == [s.prompt for s in specs]


def test_chunk_size_zero_is_single_batch() -> None:
    backend = _CountingBackend()
    specs = _make_specs(7)
    rollout_batches: list[object] = []

    results = call_generate_batch_chunked(backend, specs, rollout_batches, 0)

    assert backend.batch_sizes == [7]
    assert len(rollout_batches) == 1
    assert [r.text for r in results] == [s.prompt for s in specs]


def test_chunk_size_larger_than_specs_is_single_batch() -> None:
    backend = _CountingBackend()
    specs = _make_specs(3)
    rollout_batches: list[object] = []

    results = call_generate_batch_chunked(backend, specs, rollout_batches, 64)

    assert backend.batch_sizes == [3]
    assert len(rollout_batches) == 1
    assert len(results) == 3


def test_empty_specs_returns_empty_and_collects_nothing() -> None:
    backend = _CountingBackend()
    rollout_batches: list[object] = []

    results = call_generate_batch_chunked(backend, [], rollout_batches, 4)

    assert results == []
    assert backend.batch_sizes == []
    assert rollout_batches == []


def test_backend_without_consume_does_not_collect() -> None:
    """A fake without ``consume_last_rollout`` (e.g. smoke fakes) is a no-op
    for collection but still returns all rows in order."""

    @dataclass
    class _NoCollectBackend:
        def generate_batch(self, specs: list[GenerationSpec]) -> list[GenerationResult]:
            return [
                GenerationResult(text=spec.prompt, metadata={})
                for spec in specs
            ]

    backend = _NoCollectBackend()
    specs = _make_specs(5)
    rollout_batches: list[object] = []

    results = call_generate_batch_chunked(backend, specs, rollout_batches, 2)

    assert [r.text for r in results] == [s.prompt for s in specs]
    assert rollout_batches == []
