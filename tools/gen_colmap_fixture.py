"""Generates a synthetic mini bundle in the COLMAP binary format (for tests).
Format definition: https://colmap.github.io/format.html (Binary File Format)
"""
import struct
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "colmap_mini" / "sparse" / "0"
OUT.mkdir(parents=True, exist_ok=True)

# cameras.bin: num(uint64) | per-cam: id(i32) model_id(i32) width(u64) height(u64) params(f64 x n)
with open(OUT / "cameras.bin", "wb") as f:
    f.write(struct.pack("<Q", 2))
    f.write(struct.pack("<iiQQ", 1, 1, 1920, 1440))          # PINHOLE: fx fy cx cy
    f.write(struct.pack("<4d", 1500.0, 1510.0, 960.0, 720.0))
    f.write(struct.pack("<iiQQ", 2, 0, 640, 480))            # SIMPLE_PINHOLE: f cx cy
    f.write(struct.pack("<3d", 500.0, 320.0, 240.0))

# images.bin: num(u64) | per-img: id(i32) qvec(4d, w-first) tvec(3d) cam_id(i32)
#             name(null-terminated) num_p2d(u64) p2d(x,y f64 + id i64)*
with open(OUT / "images.bin", "wb") as f:
    f.write(struct.pack("<Q", 2))
    for img_id, name in [(1, b"frame_000.jpg"), (2, b"frame_001.jpg")]:
        f.write(struct.pack("<i", img_id))
        f.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))
        f.write(struct.pack("<3d", 0.1 * img_id, -0.2, 3.0))
        f.write(struct.pack("<i", 1))
        f.write(name + b"\x00")
        f.write(struct.pack("<Q", 2))                         # 2 points2D entries (the loader skips them)
        f.write(struct.pack("<ddq", 10.0, 20.0, -1))
        f.write(struct.pack("<ddq", 30.0, 40.0, 7))

# points3D.bin: num(u64) | per-pt: id(u64) xyz(3d) rgb(3B) error(d) track_len(u64) track(i32,i32)*
with open(OUT / "points3D.bin", "wb") as f:
    f.write(struct.pack("<Q", 3))
    pts = [(7, (1.0, 2.0, 3.0), (255, 0, 0)), (8, (-1.0, 0.5, 2.0), (0, 255, 0)),
           (9, (0.0, 0.0, 5.0), (0, 0, 255))]
    for pid, xyz, rgb in pts:
        f.write(struct.pack("<Q", pid))
        f.write(struct.pack("<3d", *xyz))
        f.write(struct.pack("<3B", *rgb))
        f.write(struct.pack("<d", 0.5))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<ii", 1, 0))

print(f"wrote {OUT}")
