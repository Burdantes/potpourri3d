#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config (edit these 2 lines)
# -----------------------------
POTPOURRI_REPO_URL="https://github.com/Burdantes/potpourri3d.git"
POTPOURRI_BRANCH="feature/custom-init-path"

# Optional: set to "0" if you don't want venv created
CREATE_VENV="${CREATE_VENV:-1}"

# Optional: choose python executable (python3 recommended)
PYTHON_BIN="${PYTHON_BIN:-python3}"

# -----------------------------
# Helpers
# -----------------------------
log() { echo -e "\033[1;32m[setup]\033[0m $*"; }
warn() { echo -e "\033[1;33m[setup]\033[0m $*"; }
die() { echo -e "\033[1;31m[setup]\033[0m $*"; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"
}

# -----------------------------
# Preconditions
# -----------------------------
need_cmd git
need_cmd "$PYTHON_BIN"

log "Using python: $("$PYTHON_BIN" --version)"

# -----------------------------
# Clone potpourri3d if needed
# -----------------------------
if [[ ! -d "potpourri3d" ]]; then
  log "Cloning potpourri3d..."
  git clone "$POTPOURRI_REPO_URL"
else
  log "potpourri3d already exists, skipping clone"
fi

cd potpourri3d

log "Checking out potpourri3d branch: $POTPOURRI_BRANCH"
git fetch --all --prune
git checkout "$POTPOURRI_BRANCH"
git pull --ff-only || warn "Could not fast-forward pull; you may have local changes."

# -----------------------------
# Submodules (THIS is what pins geometry-central)
# -----------------------------
log "Initializing/updating submodules..."
git submodule update --init --recursive

# -----------------------------
# Python environment
# -----------------------------
if [[ "$CREATE_VENV" == "1" ]]; then
  if [[ ! -d ".venv" ]]; then
    log "Creating virtual environment (.venv)..."
    "$PYTHON_BIN" -m venv .venv
  else
    log ".venv already exists"
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate
  log "Activated venv: $(which python)"
else
  warn "CREATE_VENV=0, using current python environment"
fi

log "Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

# -----------------------------
# Install editable
# -----------------------------
log "Installing potpourri3d (editable)..."
python -m pip install -e .

# -----------------------------
# Sanity check
# -----------------------------
log "Running sanity check for new binding..."
python - <<'PY'
import potpourri3d_bindings as pp3db

mgr = pp3db.EdgeFlipGeodesicsManager
if not hasattr(mgr, "find_geodesic_path_with_route"):
    raise RuntimeError("Missing find_geodesic_path_with_route on EdgeFlipGeodesicsManager")

print("OK: find_geodesic_path_with_route is available ✅")
PY

log "Done ✅"
log "Next: run testing_implementation.py inside ./potpourri3d"