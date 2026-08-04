#!/bin/sh
# SplatX.xcframework packaging (Phase 6 Task 2 — distribution artifact).
#
# Compiles SplatXCore (14 TUs) + SplatXTrainC (1 TU) + spz (3 TUs) directly into per-slice
# static libraries (xcrun clang++ — an extension of the Task 1 iOS cross-compilation
# approach), merges them with libtool, and bundles them with xcodebuild -create-xcframework.
# A single public header, splatx_train_c.h — the C API is the sole surface (SplatXCore public
# headers are C++, which Swift ClangImporter cannot parse; the first-class path for C++/Swift
# consumption is the SwiftPM source dependency — README distribution section).
#
# Slices: ios-arm64 + macos-arm64 only — **no simulator**. Rationale: the Metal training
# kernels are not viable under simulator GPU constraints (Tier/atomic/threadgroup limits, a
# driver different from real devices). Consumers who need simulator UI development use the
# SwiftPM source dependency.
#
# metallib companion contract: the XCFramework does not contain a metallib. Consumers must
# bundle the tools/build_metallib.sh output (splatx.metallib / --ios → splatx-ios.metallib)
# as an app resource and pass its path via the metallib_path argument of sx_train_run.
# KERNEL_SRC_DIR (the source-compile fallback) is deliberately defined as an invalid path so
# that no build-machine path ends up in the distributed binary — with metallib_path/
# kernel_src_dir unset and no splatx.metallib next to the executable, it fails loudly rather
# than falling back silently.
#
# Consumer link requirements (static lib, so dependencies do not propagate — matches
# Package.swift linkerSettings): -lz (spz gzip) +
# Metal/Foundation/QuartzCore (metal-cpp) + CoreGraphics/ImageIO (ImageIO.cpp) +
# libc++ (only C consumers need it explicitly — Swift/ObjC++ get it automatically).
#
# Usage: tools/build_xcframework.sh          # → build/SplatX.xcframework
#
# Verification (built into the script): 2 Info.plist slices + lipo arch + per-slice
# sx_train_run/sx_convert_run symbols (nm) + macOS slice link smoke (not executed —
# execution needs a metallib).
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(dirname -- "$SCRIPT_DIR")
BUILD_DIR="$REPO_ROOT/build"
OUT="$BUILD_DIR/SplatX.xcframework"

# Minimum deployment targets — matches Package.swift platforms (.macOS("26.0"), .iOS("26.0")).
MACOS_MIN=26.0
IOS_MIN=26.0

CXXFLAGS="-std=c++17 -O2 -DNDEBUG"
# No build-machine path exposure (carried over from Task 2 Minor): rewrites absolute paths
# in __FILE__/debug info to repo-relative ones — distribution reproducibility + path privacy.
# (Note: same principle as the KERNEL_SRC_DIR invalidation — no build-machine paths in
# distributed binaries.)
CXXFLAGS="$CXXFLAGS -ffile-prefix-map=$REPO_ROOT=."

command -v xcrun >/dev/null 2>&1 \
    || { echo "error: xcrun not found (Xcode required)" >&2; exit 1; }
command -v xcodebuild >/dev/null 2>&1 \
    || { echo "error: xcodebuild not found (Xcode required)" >&2; exit 1; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/splatx-xcf.XXXXXX")
# -create-xcframework requires -output to end in .xcframework and rejects an existing path
# — build under a temporary name, then swap.
TMP_OUT="$BUILD_DIR/SplatX.tmp.$$.xcframework"
trap 'rm -rf "$WORK" "$TMP_OUT"' EXIT

# Compile one slice + libtool merge: build_slice <sdk> <target> <objdir>
build_slice() {
    sdk=$1; target=$2; objdir=$3
    mkdir -p "$objdir"
    echo "== slice: $target (sdk $sdk)"

    # SplatXCore — include paths matching Package.swift cxxSettings
    # (metal-cpp, KernelConstants.h etc. from SplatXKernelSrc, spz public headers).
    for f in "$REPO_ROOT/Sources/SplatXCore"/*.cpp; do
        base=$(basename "$f" .cpp)
        xcrun -sdk "$sdk" clang++ $CXXFLAGS -target "$target" -c "$f" \
            -o "$objdir/core_$base.o" \
            -I "$REPO_ROOT/Sources/SplatXCore/include" \
            -I "$REPO_ROOT/ThirdParty/metal-cpp" \
            -I "$REPO_ROOT/Sources/SplatXKernelSrc" \
            -I "$REPO_ROOT/ThirdParty/spz/src/cc" \
            || { echo "error: compile failed: $f ($target)" >&2; exit 1; }
    done

    # SplatXTrainC — KERNEL_SRC_DIR is a deliberately invalid path (see the header comment:
    # no build-machine paths in distributed binaries — consumers inject at runtime via the
    # metallib_path or kernel_src_dir argument; reaching the fallback fails loudly).
    xcrun -sdk "$sdk" clang++ $CXXFLAGS -target "$target" \
        -c "$REPO_ROOT/Sources/SplatXTrainC/TrainC.cpp" \
        -o "$objdir/trainc_TrainC.o" \
        -I "$REPO_ROOT/Sources/SplatXTrainC/include" \
        -I "$REPO_ROOT/Sources/SplatXCore/include" \
        -DKERNEL_SRC_DIR='"/nonexistent/SplatXKernelSrc-not-bundled"' \
        || { echo "error: compile failed: TrainC.cpp ($target)" >&2; exit 1; }

    # spz (vendored v2.1.0 — gzip single stream, zlib only; linking is the consumer's job, -lz).
    # splat-c-types.cc is "// Empty." upstream — the libtool "no symbols" warning is normal.
    for f in "$REPO_ROOT/ThirdParty/spz/src/cc"/*.cc; do
        base=$(basename "$f" .cc)
        xcrun -sdk "$sdk" clang++ $CXXFLAGS -target "$target" -c "$f" \
            -o "$objdir/spz_$base.o" \
            -I "$REPO_ROOT/ThirdParty/spz/src/cc" \
            || { echo "error: compile failed: $f ($target)" >&2; exit 1; }
    done

    xcrun libtool -static -o "$objdir/libSplatX.a" "$objdir"/*.o \
        || { echo "error: libtool merge failed ($target)" >&2; exit 1; }
}

build_slice iphoneos "arm64-apple-ios$IOS_MIN"    "$WORK/ios"
build_slice macosx   "arm64-apple-macos$MACOS_MIN" "$WORK/macos"

# Public header staging — the C API is the sole surface
mkdir -p "$WORK/Headers"
cp "$REPO_ROOT/Sources/SplatXTrainC/include/splatx_train_c.h" "$WORK/Headers/"

# Create the xcframework (build at a temporary path, then swap — -output rejects an existing path)
mkdir -p "$BUILD_DIR"
xcodebuild -create-xcframework \
    -library "$WORK/ios/libSplatX.a"   -headers "$WORK/Headers" \
    -library "$WORK/macos/libSplatX.a" -headers "$WORK/Headers" \
    -output "$TMP_OUT" \
    || { echo "error: xcodebuild -create-xcframework failed" >&2; exit 1; }
rm -rf "$OUT"
mv "$TMP_OUT" "$OUT"

# ---- Verification ----
fail() { echo "error: $1" >&2; exit 1; }

# 1) 2 Info.plist slices (ios-arm64 + macos-arm64 — no other slices)
PLIST_DUMP=$(plutil -p "$OUT/Info.plist")
for id in ios-arm64 macos-arm64; do
    echo "$PLIST_DUMP" | grep -q "\"LibraryIdentifier\" => \"$id\"" \
        || fail "slice $id missing from Info.plist"
    [ -f "$OUT/$id/libSplatX.a" ] || fail "$id/libSplatX.a missing"
    [ -f "$OUT/$id/Headers/splatx_train_c.h" ] || fail "$id public header missing"
done
n_slices=$(echo "$PLIST_DUMP" | grep -c '"LibraryIdentifier"')
[ "$n_slices" -eq 2 ] || fail "slice count $n_slices != 2"

# 2) Architecture + sx_* symbols (per slice)
for id in ios-arm64 macos-arm64; do
    lib="$OUT/$id/libSplatX.a"
    archs=$(lipo -archs "$lib")
    [ "$archs" = "arm64" ] || fail "$id arch '$archs' != arm64"
    for sym in _sx_train_run _sx_convert_run; do
        nm -gU "$lib" 2>/dev/null | grep -q "T $sym\$" \
            || fail "symbol $sym missing from $id"
    done
    echo "verified: $id (arm64, sx_train_run + sx_convert_run)"
done

# 3) macOS slice link smoke — a C consumer links the sx_* symbols from the header alone
#    (not executed — execution needs a metallib/data. Demonstrates the consumer link
#    requirements: -lz + 5 frameworks + libc++ (clang++ link driver)).
cat > "$WORK/smoke.c" <<'EOF'
#include <stdio.h>
#include "splatx_train_c.h"
int main(void) {
    printf("%p %p\n", (void*)&sx_train_run, (void*)&sx_convert_run);
    return 0;
}
EOF
xcrun -sdk macosx clang -target "arm64-apple-macos$MACOS_MIN" \
    -c "$WORK/smoke.c" -o "$WORK/smoke.o" -I "$OUT/macos-arm64/Headers" \
    || fail "link smoke: smoke.c compile failed"
xcrun -sdk macosx clang++ -target "arm64-apple-macos$MACOS_MIN" \
    "$WORK/smoke.o" "$OUT/macos-arm64/libSplatX.a" -o "$WORK/smoke" \
    -lz -framework Metal -framework Foundation -framework QuartzCore \
    -framework CoreGraphics -framework ImageIO \
    || fail "link smoke: macos-arm64 link failed"
echo "verified: macos-arm64 link smoke (C consumer, -lz + frameworks)"

echo "built: $OUT"
