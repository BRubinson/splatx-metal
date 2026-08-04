#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace splatx {

struct ColmapCamera { int32_t id; int32_t model_id; int64_t width, height; std::vector<double> params; float fx, fy, cx, cy; };
struct ColmapImage  { int32_t id; double qvec[4]; double tvec[3]; int32_t camera_id; std::string name; }; // qvec: w-first, world→cam
struct ColmapPoint3D { double xyz[3]; uint8_t rgb[3]; };
struct ColmapBundle {
    std::vector<ColmapCamera> cameras;
    std::vector<ColmapImage> images;
    std::vector<ColmapPoint3D> points;
};
// bin/txt auto-detection: if cameras.bin exists, use bin (preferred); if it is absent and
// cameras.txt exists, use txt (the iOS capture app bundle — txt writer).
// If neither exists, throw.
ColmapBundle load_colmap_bundle(const std::string& sparse_dir); // e.g. "path/sparse/0"

} // namespace splatx
