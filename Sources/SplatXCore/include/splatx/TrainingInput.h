// TrainingInput — the trainer's input data contract.
// The common representation of the COLMAP on-disk bundle and the iOS capture app's in-memory
// input (ARKit poses + LiDAR seed).
// Phase 0 defines only the structs and the COLMAP conversion (pixel loading is Phase 1, the
// in-memory provider is Phase 6).
#pragma once
#include "splatx/ColmapLoader.h"
#include <cstdint>
#include <string>
#include <vector>

namespace splatx {

struct TrainingCamera {
    float viewmat[16];       // world→cam 4x4 row-major (composed from COLMAP qvec/tvec)
    float K[9];              // intrinsics
    int width = 0, height = 0;
    std::string image_path;  // image file path (lazy loading)
    // Depth sidecar beside the image folder (DepthSupervision.h): depths/<stem>.png and
    // depths/<stem>.conf.png, each empty when the file does not exist.
    std::string depth_path;
    std::string confidence_path;
    // Region mask sidecar (RegionMask.h): masks/<stem>.png beside the image folder, empty when the
    // file does not exist (the view is then unmasked).
    std::string mask_path;
};

struct TrainingInput {
    std::vector<TrainingCamera> cameras;
    std::vector<float> seed_points; // [M,3] (COLMAP points3D or LiDAR seed)
    std::vector<uint8_t> seed_rgb;  // [M,3]
};

// Also resolves each image's depth sidecar under <parent of images_dir>/depths/ and its region
// mask under <parent of images_dir>/masks/.
TrainingInput training_input_from_colmap(const ColmapBundle& bundle, const std::string& images_dir);

// Resolves the image folder for a given factor (Integration Task 2 — the iOS capture app bundle
// has no pre-resized images_N folder). The convention:
//   factor 1                → {<data>/images, false}
//   images_{factor} exists  → {<data>/images_{factor}, false}  (unchanged behavior)
//   absent                  → {<data>/images, true}            (falls back to downscaling by
//                             factor at load time — load_image_rgb_f32_downscaled, dimension
//                             convention downscaled_dim)
// If neither exists, throw (an explicit error — this is not deferred to the first decode).
struct ImagesDirChoice {
    std::string dir;                // image folder path
    bool downscale_on_load = false; // true = the images_{factor}-absent fallback is engaged
};
ImagesDirChoice resolve_images_dir(const std::string& data_dir, int factor);

} // namespace splatx
