// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983.
// Reimplemented for Metal/C++.
#pragma once

namespace splatx {

struct Vec3 { float x, y, z; };
struct Mat3 { float m[9]; }; // row-major

Mat3 quat_to_rotmat(const float q[4]);                       // q: w-first, normalized internally
Mat3 quat_scale_to_covar(const float q[4], const float s[3]); // (R·diag(s))·(R·diag(s))ᵀ
Mat3 mat3_mul(const Mat3& a, const Mat3& b);
Mat3 mat3_transpose(const Mat3& a);

} // namespace splatx
