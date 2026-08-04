// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983.
// Reimplemented for Metal/C++.
// Reference: _quat_to_rotmat and _quat_scale_to_covar_preci in gsplat/cuda/_math.py
#include "splatx/Math.h"
#include <cmath>

namespace splatx {

Mat3 quat_to_rotmat(const float q[4]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    float n = std::sqrt(w * w + x * x + y * y + z * z);
    if (n > 0.0f) { w /= n; x /= n; y /= n; z /= n; }
    Mat3 R;
    R.m[0] = 1 - 2 * (y * y + z * z); R.m[1] = 2 * (x * y - w * z); R.m[2] = 2 * (x * z + w * y);
    R.m[3] = 2 * (x * y + w * z); R.m[4] = 1 - 2 * (x * x + z * z); R.m[5] = 2 * (y * z - w * x);
    R.m[6] = 2 * (x * z - w * y); R.m[7] = 2 * (y * z + w * x); R.m[8] = 1 - 2 * (x * x + y * y);
    return R;
}

Mat3 mat3_mul(const Mat3& a, const Mat3& b) {
    Mat3 c;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float s = 0;
            for (int k = 0; k < 3; k++) s += a.m[i * 3 + k] * b.m[k * 3 + j];
            c.m[i * 3 + j] = s;
        }
    return c;
}

Mat3 mat3_transpose(const Mat3& a) {
    Mat3 t;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) t.m[i * 3 + j] = a.m[j * 3 + i];
    return t;
}

Mat3 quat_scale_to_covar(const float q[4], const float s[3]) {
    Mat3 R = quat_to_rotmat(q);
    Mat3 M = R;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) M.m[i * 3 + j] *= s[j]; // M = R * diag(s)
    return mat3_mul(M, mat3_transpose(M));
}

} // namespace splatx
