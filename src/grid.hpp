#pragma once

#include <cmath>
#include <cstdint>
#include <array>
#include <stdexcept>

static constexpr double two_pi = 6.283185307179586476925286766559;

// Regular grid and the box it covers. Nothing here is specific to cosmology: lbox carries whatever
// length unit the caller uses, and only the ratios between the axes enter the kernels. The overall
// scale is a final factor on k and on the power.
struct Grid3D {
  std::array<int, 3> n{};
  std::array<double, 3> lbox{};

  // number of complex modes kept by a real-to-complex transform along the last axis
  inline int nz_c() const
  {
    return n[2] / 2 + 1;
  }

  inline int64_t size() const
  {
    return (int64_t)n[0] * (int64_t)n[1] * (int64_t)n[2];
  }

  inline int64_t csize() const
  {
    return (int64_t)n[0] * (int64_t)n[1] * (int64_t)nz_c();
  }

  // fundamental mode of an axis
  inline double kf(const int axis) const
  {
    return two_pi / lbox[axis];
  }

  // Nyquist wavenumber of an axis
  inline double knyq(const int axis) const
  {
    return 0.5 * (double)n[axis] * kf(axis);
  }

  inline double volume() const
  {
    return lbox[0] * lbox[1] * lbox[2];
  }
};

static inline Grid3D make_grid(const std::array<int, 3> &n, const std::array<double, 3> &lbox)
{
  for(int j = 0; j < 3; j++) {
    if(n[j] <= 0) throw std::runtime_error("mesh shape must be positive");
    if(!std::isfinite(lbox[j]) || lbox[j] <= 0.0) throw std::runtime_error("lbox must be positive and finite");
  }

  Grid3D grid;
  grid.n = n;
  grid.lbox = lbox;
  return grid;
}

// signed integer wavenumber of a real-to-complex layout along a full axis
static inline int wave_index(const int i, const int n)
{
  return (i <= n / 2) ? i : i - n;
}

// The kz = 0 plane, and the kz = Nyquist plane of an even mesh, hold each mode and its conjugate.
// Their redundant half is skipped, so nmodes counts independent modes, as sqrt(2 / nmodes) expects.
static inline bool is_conjugate_plane(const int iz, const int nz)
{
  return (iz == 0) || ((nz % 2 == 0) && (iz == nz / 2));
}

// Whether a mode of a conjugate plane is the redundant copy of one already visited. Kept as a
// per-(ix, iy) test so the innermost loop only sees a boolean.
static inline bool is_redundant_in_plane(const int mx, const int my, const int nx, const int ny)
{
  if(mx < 0) return true;
  if(mx == 0 || ((nx % 2 == 0) && (mx == nx / 2))) return my < 0;
  return false;
}
