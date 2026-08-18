# Sourced by every repro/*.sh entrypoint. Not run on its own.
#
# Sets:
#   REPO_ROOT  — repo root
#   PYTHONPATH — repo scripts/ + upstream model sources (needed only to re-create)
# Discovers the three upstream models as sibling clones (see models/SETUP.md),
# or honor MOLMINER_ROOT / GDSS_ROOT / HGRAPH2GRAPH_ROOT if exported.
# Analyzing (plotting only) needs none of the model sources; never hard-fails on them.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$REPO_ROOT/scripts:${PYTHONPATH:-}"

find_root () {  # $1=var-already-set $2=marker subdir, $3..=candidate dirs
  local marker="$1"; shift
  for c in "$@"; do
    if [ -d "$c/$marker" ]; then (cd "$c" && pwd); return 0; fi
  done
  return 1
}

MOLMINER_ROOT="${MOLMINER_ROOT:-$(find_root molminer "$REPO_ROOT/../molminer" "$HOME/src/molminer" || true)}"
[ -n "$MOLMINER_ROOT" ] && export MOLMINER_ROOT && export PYTHONPATH="$MOLMINER_ROOT:$PYTHONPATH"

GDSS_ROOT="${GDSS_ROOT:-$(find_root utils "$REPO_ROOT/../GDSS" "$HOME/src/GDSS" || true)}"
[ -n "$GDSS_ROOT" ] && export GDSS_ROOT

HGRAPH2GRAPH_ROOT="${HGRAPH2GRAPH_ROOT:-$(find_root hgraph "$REPO_ROOT/../hgraph2graph" "$HOME/src/hgraph2graph" || true)}"
[ -n "$HGRAPH2GRAPH_ROOT" ] && export HGRAPH2GRAPH_ROOT && export PYTHONPATH="$HGRAPH2GRAPH_ROOT:$PYTHONPATH"

# Activate a conda env by name. Sources conda.sh first (a non-interactive shell
# has no `conda` on PATH), discovering it via $CONDA_EXE or common install dirs.
# Override by exporting CONDA_SH=/path/to/etc/profile.d/conda.sh.
activate_env () {
  local env="$1"
  if ! type conda >/dev/null 2>&1; then
    local sh="${CONDA_SH:-}"
    [ -z "$sh" ] && [ -n "${CONDA_EXE:-}" ] && sh="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
    if [ -z "$sh" ]; then
      for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
        [ -f "$c/etc/profile.d/conda.sh" ] && sh="$c/etc/profile.d/conda.sh" && break
      done
    fi
    # shellcheck disable=SC1090
    [ -n "$sh" ] && [ -f "$sh" ] && source "$sh"
  fi
  type conda >/dev/null 2>&1 || { echo "FATAL: conda not found; set CONDA_SH or activate '$env' manually" >&2; return 1; }
  conda activate "$env"
}
