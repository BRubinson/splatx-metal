#!/bin/zsh
# App Store screenshot composition: places a raw capture onto the brand frame (2880×1800).
# Usage:   ./compose.sh <capture.png> <headline> <subline> <output.png>
# Example: ./compose.sh ~/Desktop/train.png "Train on your Mac" "3D Gaussian Splatting, native Metal" out/01-train.png
set -euo pipefail
SRC="$1"; HEADLINE="$2"; SUBLINE="$3"; OUT="$4"
DIR="$(cd "$(dirname "$0")" && pwd)"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TMP="$(mktemp -d)/frame.html"

python3 - "$DIR/frame-template.html" "$SRC" "$HEADLINE" "$SUBLINE" "$TMP" <<'EOF'
import sys, pathlib, html
tpl, src, head, sub, out = sys.argv[1:6]
s = pathlib.Path(tpl).read_text()
s = s.replace('{{IMAGE}}', pathlib.Path(src).resolve().as_uri())
s = s.replace('{{HEADLINE}}', html.escape(head)).replace('{{SUBLINE}}', html.escape(sub))
pathlib.Path(out).write_text(s)
EOF

mkdir -p "$(dirname "$OUT")"
"$CHROME" --headless=new --disable-gpu --force-device-scale-factor=1 \
  --window-size=2880,1800 --screenshot="$OUT" "file://$TMP" 2>/dev/null
echo "wrote $OUT (2880x1800)"
