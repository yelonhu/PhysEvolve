#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/setup_verl.sh [--target PATH] [--repo-url URL] [--force]

Recreate the PhysEvolve-compatible veRL checkout from the pinned upstream commit
and the patch queue stored in third_party/verl/patches.

Defaults:
  --target ../verl
  --repo-url https://github.com/verl-project/verl.git

Options:
  --force                 Reset an existing target checkout before applying patches.
                          Without this flag, the script refuses dirty checkouts.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERL_META_DIR="${REPO_ROOT}/third_party/verl"
PATCH_DIR="${VERL_META_DIR}/patches"
BASE_COMMIT="$(tr -d '[:space:]' < "${VERL_META_DIR}/base_commit.txt")"

TARGET="${REPO_ROOT}/../verl"
REPO_URL="${VERL_REPO_URL:-https://github.com/verl-project/verl.git}"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"
      shift 2
      ;;
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "${TARGET}/.git" ]]; then
  mkdir -p "$(dirname "${TARGET}")"
  git clone "${REPO_URL}" "${TARGET}"
fi

cd "${TARGET}"

if [[ "${FORCE}" -eq 1 ]]; then
  git reset --hard
  git clean -fd
elif [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to modify dirty veRL checkout at ${TARGET}." >&2
  echo "Commit/stash changes or rerun with --force to discard them." >&2
  exit 1
fi

git fetch origin "${BASE_COMMIT}" || true
git checkout --detach "${BASE_COMMIT}"

if [[ "${FORCE}" -eq 1 ]]; then
  git reset --hard "${BASE_COMMIT}"
  git clean -fd
fi

PATCHES=()
for patch in "${PATCH_DIR}"/*.patch; do
  [[ -e "${patch}" ]] || continue
  name="$(basename "${patch}")"
  PATCHES+=("${patch}")
done

if [[ "${#PATCHES[@]}" -eq 0 ]]; then
  echo "No veRL patch files found under ${PATCH_DIR}" >&2
  exit 1
fi

for patch in "${PATCHES[@]}"; do
  git apply --check "${patch}"
  git apply "${patch}"
done

echo "PhysEvolve-compatible veRL prepared at ${TARGET}"
echo "Base commit: ${BASE_COMMIT}"
echo "Applied patches:"
printf '  %s\n' "${PATCHES[@]##*/}"
