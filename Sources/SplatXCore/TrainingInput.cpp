// Trainer input data contract — converts a COLMAP bundle into TrainingInput. Written from scratch.
#include "splatx/TrainingInput.h"
#include "splatx/Math.h"
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

namespace splatx {

TrainingInput training_input_from_colmap(const ColmapBundle& b, const std::string& images_dir) {
    std::unordered_map<int32_t, const ColmapCamera*> cam_by_id;
    for (const auto& c : b.cameras) cam_by_id[c.id] = &c;

    TrainingInput ti;
    ti.cameras.reserve(b.images.size());
    for (const auto& im : b.images) {
        auto it = cam_by_id.find(im.camera_id);
        if (it == cam_by_id.end())
            throw std::runtime_error("training_input: image references unknown camera " + std::to_string(im.camera_id));
        const ColmapCamera& cam = *it->second;

        TrainingCamera tc;
        float q[4] = {static_cast<float>(im.qvec[0]), static_cast<float>(im.qvec[1]),
                      static_cast<float>(im.qvec[2]), static_cast<float>(im.qvec[3])};
        Mat3 R = quat_to_rotmat(q); // COLMAP qvec is w-first, a world→cam rotation
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) tc.viewmat[i * 4 + j] = R.m[i * 3 + j];
        for (int i = 0; i < 3; i++) tc.viewmat[i * 4 + 3] = static_cast<float>(im.tvec[i]);
        tc.viewmat[12] = 0; tc.viewmat[13] = 0; tc.viewmat[14] = 0; tc.viewmat[15] = 1;

        tc.K[0] = cam.fx; tc.K[1] = 0;      tc.K[2] = cam.cx;
        tc.K[3] = 0;      tc.K[4] = cam.fy; tc.K[5] = cam.cy;
        tc.K[6] = 0;      tc.K[7] = 0;      tc.K[8] = 1;
        tc.width = static_cast<int>(cam.width);
        tc.height = static_cast<int>(cam.height);
        tc.image_path = images_dir + "/" + im.name;
        ti.cameras.push_back(std::move(tc));
    }

    ti.seed_points.reserve(b.points.size() * 3);
    ti.seed_rgb.reserve(b.points.size() * 3);
    for (const auto& p : b.points) {
        for (int k = 0; k < 3; k++) ti.seed_points.push_back(static_cast<float>(p.xyz[k]));
        for (int k = 0; k < 3; k++) ti.seed_rgb.push_back(p.rgb[k]);
    }
    return ti;
}

ImagesDirChoice resolve_images_dir(const std::string& data_dir, int factor) {
    namespace fs = std::filesystem;
    const std::string originals = data_dir + "/images";
    std::error_code ec; // an is_directory error (permissions etc.) counts as "absent" — it converges on the explicit error below
    if (factor <= 1) return {originals, false};
    const std::string scaled = data_dir + "/images_" + std::to_string(factor);
    if (fs::is_directory(scaled, ec)) return {scaled, false}; // exactly the previous behavior (absolute precedence)
    if (fs::is_directory(originals, ec)) return {originals, true}; // fallback: downscale at load time
    throw std::runtime_error("resolve_images_dir: neither " + scaled + " nor " + originals +
                             " exists");
}

} // namespace splatx
