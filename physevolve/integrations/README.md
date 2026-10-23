# PhysEvolve Integrations

This package isolates PhysEvolve from external runtimes (vLLM, Hugging Face
processors). The active training path lives in `physevolve/training/`
and uses this package only for video payload conversion.

## Files

| File | Role |
| --- | --- |
| `video_bridge.py` | Canonical `SimState -> (frames, metadata)` conversion for Qwen3-VL multimodal rollouts. Sole owner of the vLLM video metadata contract. |

## Video Contract

`video_bridge.py` emits contiguous `uint8` frames plus Qwen3-VL metadata
containing:

```text
total_num_frames, fps, width, height, duration, frames_indices, do_sample_frames
```

`do_sample_frames` is always `False` because task builders choose the frame
slice before the payload reaches vLLM.
