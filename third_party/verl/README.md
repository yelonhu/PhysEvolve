# veRL Patch Queue

PhysEvolve does not vendor veRL.  This directory records how to reconstruct the
PhysEvolve-compatible veRL checkout used by the reported runs.

- `base_commit.txt` pins the upstream veRL commit.
- `patches/*.patch` are applied in lexical order.
- `scripts/setup_verl.sh` clones or resets a sibling veRL checkout and applies
  this queue.

For details, see [architecture and patches](../../docs/architecture.md#verl-integration).
