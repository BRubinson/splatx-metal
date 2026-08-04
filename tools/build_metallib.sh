#!/bin/sh
# Offline compilation of splatx.metallib (Phase 2 Task 2 — the canonical v0.1 relocatable
# distribution artifact).
#
# Sources/SplatXKernelSrc/*.metal → (xcrun metal -c, one .air per file) → xcrun metallib link.
# Unlike runtime source compilation (MetalBackend fallback tier c — shared headers prepended +
# local includes stripped, then concatenated), offline compilation resolves each .metal's local
# includes as-is via -I. The sources are self-contained (each file includes the headers it
# needs) and header helpers are static inline (D5 — no duplicate symbols), so separate-TU
# compilation + metallib linking produces no symbol collisions.
# function constants stay unspecialized in the metallib — they are specialized at pipeline
# creation (newFunction(name:constantValues:)), so behavior matches the runtime compile path.
#
# Usage: tools/build_metallib.sh [--ios] [-o <output.metallib>]
#   default (macOS): -sdk macosx, output <repo>/build/splatx.metallib
#   --ios      : -sdk iphoneos, output <repo>/build/splatx-ios.metallib (Phase 6 Task 1 —
#                formalizes the iOS variant, previously comment-reference only, as an option.
#                Same air→metallib procedure, only the SDK changes. The metallib is a resource
#                file — no code signing needed. function constants likewise stay
#                unspecialized — specialized at pipeline creation)
#
# Full-suite metallib verification (exercises every kernel entry + all function constant
# specializations):
#   tools/build_metallib.sh && SPLATX_TEST_METALLIB=build/splatx.metallib swift run splatx-core-tests
#
# MSL 3.2 — full recompilation on every run (no incremental cache — CI reproducibility).
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(dirname -- "$SCRIPT_DIR")
SRC_DIR="$REPO_ROOT/Sources/SplatXKernelSrc"
SDK=macosx
OUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --ios) SDK=iphoneos ;;
        -o)
            [ $# -ge 2 ] || { echo "usage: $0 [--ios] [-o output.metallib]" >&2; exit 2; }
            OUT=$2; shift ;;
        *) echo "usage: $0 [--ios] [-o output.metallib]" >&2; exit 2 ;;
    esac
    shift
done

# Per-SDK default output when -o is not given (macOS default behavior unchanged).
if [ -z "$OUT" ]; then
    if [ "$SDK" = "iphoneos" ]; then
        OUT="$REPO_ROOT/build/splatx-ios.metallib"
    else
        OUT="$REPO_ROOT/build/splatx.metallib"
    fi
fi

[ -d "$SRC_DIR" ] || { echo "error: kernel source dir not found: $SRC_DIR" >&2; exit 1; }
command -v xcrun >/dev/null 2>&1 \
    || { echo "error: xcrun not found (Xcode / Command Line Tools required)" >&2; exit 1; }

AIR_DIR=$(mktemp -d "${TMPDIR:-/tmp}/splatx-air.XXXXXX")
trap 'rm -rf "$AIR_DIR"' EXIT

found=0
for f in "$SRC_DIR"/*.metal; do
    [ -e "$f" ] || break # guard against an unmatched glob (sh keeps the literal)
    found=1
    base=$(basename "$f" .metal)
    xcrun -sdk "$SDK" metal -c "$f" -o "$AIR_DIR/$base.air" -I "$SRC_DIR" -std=metal3.2 \
        || { echo "error: metal compile failed: $f" >&2; exit 1; }
done
[ "$found" -eq 1 ] || { echo "error: no .metal sources in $SRC_DIR" >&2; exit 1; }

mkdir -p "$(dirname -- "$OUT")"
# Output atomicity: link into a temp file in the same directory, then mv (a same-filesystem
# rename is atomic — concurrent runs / consumers loading it never see a truncated metallib).
TMP_OUT="$OUT.tmp.$$"
trap 'rm -rf "$AIR_DIR" "$TMP_OUT"' EXIT
xcrun -sdk "$SDK" metallib "$AIR_DIR"/*.air -o "$TMP_OUT" \
    || { echo "error: metallib link failed" >&2; exit 1; }
mv -f "$TMP_OUT" "$OUT"
echo "built: $OUT"
