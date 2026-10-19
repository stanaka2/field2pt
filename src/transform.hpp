#pragma once

#include <cmath>
#include <cstdint>
#include <complex>

#include <omp.h>

#include "grid.hpp"

// Adds a field built from the particles moved by a fraction a of a cell into a running sum,
// acc += exp(i k.s) d, or turns acc itself when d == nullptr. Over a set of shifts image n keeps
// exp(-2 pi i n.a) and cancels unless every shift keeps it in phase. The 1/N goes on the power.
template <typename T>
static void interlace_add(std::complex<T> *acc, const std::complex<T> *d, const Grid3D &grid, const double a[3],
                          const int nthreads)
{
  const int nx = grid.n[0];
  const int ny = grid.n[1];
  const int nz = grid.n[2];
  const int nzc = grid.nz_c();

  if(nthreads > 0) omp_set_num_threads(nthreads);

#pragma omp parallel for collapse(2) schedule(static)
  for(int ix = 0; ix < nx; ix++) {
    for(int iy = 0; iy < ny; iy++) {

      const double px = 2.0 * M_PI * a[0] * (double)wave_index(ix, nx) / (double)nx;
      const double py = 2.0 * M_PI * a[1] * (double)wave_index(iy, ny) / (double)ny;

      const int64_t base = (int64_t)nzc * ((int64_t)iy + (int64_t)ny * (int64_t)ix);

      for(int iz = 0; iz < nzc; iz++) {

        const double phase = px + py + 2.0 * M_PI * a[2] * (double)iz / (double)nz;
        const double c = std::cos(phase);
        const double s = std::sin(phase);

        const int64_t idx = base + iz;
        const std::complex<T> b = (d != nullptr) ? d[idx] : acc[idx];
        const double a_re = (d != nullptr) ? (double)acc[idx].real() : 0.0;
        const double a_im = (d != nullptr) ? (double)acc[idx].imag() : 0.0;
        const double b_re = (double)b.real();
        const double b_im = (double)b.imag();

        acc[idx] = std::complex<T>((T)(a_re + c * b_re - s * b_im), (T)(a_im + c * b_im + s * b_re));
      }
    }
  } // ix,iy,iz
}

// Adds i k_axis d, one term of the divergence of a vector field, into a running sum, or starts the
// sum with it when first is set. Every entry is visited: this rewrites the array.
template <typename T>
static void divergence_add(std::complex<T> *acc, const std::complex<T> *d, const Grid3D &grid, const int axis,
                           const bool first, const int nthreads)
{
  const int nx = grid.n[0];
  const int ny = grid.n[1];
  const int nzc = grid.nz_c();
  const double kf = grid.kf(axis);

  if(nthreads > 0) omp_set_num_threads(nthreads);

#pragma omp parallel for collapse(2) schedule(static)
  for(int ix = 0; ix < nx; ix++) {
    for(int iy = 0; iy < ny; iy++) {

      const int64_t base = (int64_t)nzc * ((int64_t)iy + (int64_t)ny * (int64_t)ix);

      for(int iz = 0; iz < nzc; iz++) {

        const int m = (axis == 0) ? wave_index(ix, nx) : ((axis == 1) ? wave_index(iy, ny) : iz);
        const double ka = (double)m * kf;

        const int64_t idx = base + iz;
        const double a_re = first ? 0.0 : (double)acc[idx].real();
        const double a_im = first ? 0.0 : (double)acc[idx].imag();
        const double b_re = (double)d[idx].real();
        const double b_im = (double)d[idx].imag();

        acc[idx] = std::complex<T>((T)(a_re - ka * b_im), (T)(a_im + ka * b_re)); // += i k b
      }
    }
  } // ix,iy,iz
}

static inline double boxcar_axis_window(const double x)
{
  return x == 0.0 ? 1.0 : std::sin(x) / x;
}

// Kind 0: Gaussian exp(-k^2 R^2 / 2). Kind 1: spherical top-hat
// 3 [sin(kR) - kR cos(kR)] / (kR)^3. Kind 2: boxcar prod_i sinc(k_i R),
// with full width 2R along each coordinate axis.
template <typename T>
static void smooth_spectrum(std::complex<T> *d, const Grid3D &grid, const double radius, const int kind,
                            const int nthreads)
{
  const int nx = grid.n[0];
  const int ny = grid.n[1];
  const int nzc = grid.nz_c();

  const double kfx = grid.kf(0);
  const double kfy = grid.kf(1);
  const double kfz = grid.kf(2);

  if(nthreads > 0) omp_set_num_threads(nthreads);

#pragma omp parallel for collapse(2) schedule(static)
  for(int ix = 0; ix < nx; ix++) {
    for(int iy = 0; iy < ny; iy++) {

      const double kx = (double)wave_index(ix, nx) * kfx;
      const double ky = (double)wave_index(iy, ny) * kfy;
      const double kxy2 = kx * kx + ky * ky;

      const int64_t base = (int64_t)nzc * ((int64_t)iy + (int64_t)ny * (int64_t)ix);

      for(int iz = 0; iz < nzc; iz++) {

        const double kz = (double)iz * kfz;
        const double k = std::sqrt(kxy2 + kz * kz);
        const double x = k * radius;

        double w;
        if(kind == 0) {
          w = std::exp(-0.5 * x * x);
        } else if(kind == 1) {
          // the series limit avoids the 0/0 at k = 0 and the cancellation just above it
          w = (x < 1e-4) ? (1.0 - 0.1 * x * x) : 3.0 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
        } else {
          w = boxcar_axis_window(kx * radius) * boxcar_axis_window(ky * radius) * boxcar_axis_window(kz * radius);
        }

        const int64_t idx = base + iz;
        d[idx] = std::complex<T>((T)((double)d[idx].real() * w), (T)((double)d[idx].imag() * w));
      }
    }
  } // ix,iy,iz
}
