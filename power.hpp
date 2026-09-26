#pragma once

#include <cmath>
#include <cstdint>
#include <complex>
#include <vector>

#include <omp.h>

#include "grid.hpp"
#include "binning.hpp"
#include "window.hpp"

// Binned power of one field, d2 == nullptr, or of two, taking Re (part 0) or Im (part 1) of
// d1 conj(d2): Re carries the even multipoles and Im the odd ones. KPOW = 2 weights each mode by
// k^2, for the vorticity. Each thread bins into its own arrays; the caller applies the volume norm.
template <int KPOW, typename T>
static void accumulate_power(const std::complex<T> *d1, const std::complex<T> *d2, const Grid3D &grid, const int method,
                             const int part, const BinSpec &bins, const int nells, const int *ells, const int los,
                             double *sum_p, double *sum_k, double *nmodes, const int nthreads)
{
  const int nx = grid.n[0];
  const int ny = grid.n[1];
  const int nz = grid.n[2];
  const int nzc = grid.nz_c();
  const int nbin = bins.nbin;

  const double kfx = grid.kf(0);
  const double kfy = grid.kf(1);
  const double kfz = grid.kf(2);

  if(nthreads > 0) omp_set_num_threads(nthreads);

#pragma omp parallel
  {
    std::vector<double> p_thr((size_t)nbin * nells, 0.0);
    std::vector<double> k_thr(nbin, 0.0);
    std::vector<double> n_thr(nbin, 0.0);

#pragma omp for collapse(2) schedule(static)
    for(int ix = 0; ix < nx; ix++) {
      for(int iy = 0; iy < ny; iy++) {

        const int mx = wave_index(ix, nx);
        const int my = wave_index(iy, ny);

        const double kx = (double)mx * kfx;
        const double ky = (double)my * kfy;
        const double kxy2 = kx * kx + ky * ky;

        const double wx = sinc_ratio(mx, nx);
        const double wy = sinc_ratio(my, ny);

        // the conjugate planes store each mode twice; this (ix, iy) holds the redundant copy
        const bool drop_in_plane = is_redundant_in_plane(mx, my, nx, ny);

        const int64_t base = (int64_t)nzc * ((int64_t)iy + (int64_t)ny * (int64_t)ix);

        for(int iz = 0; iz < nzc; iz++) {

          if(drop_in_plane && is_conjugate_plane(iz, nz)) continue;

          const double kz = (double)iz * kfz;
          const double k2 = kxy2 + kz * kz;
          const double k = std::sqrt(k2);

          const int ib = bin_index(k, bins);
          if(ib < 0) continue;

          const int64_t idx = base + iz;
          double pw;
          if(d2 == nullptr) {
            const double re = (double)d1[idx].real();
            const double ig = (double)d1[idx].imag();
            pw = re * re + ig * ig;
          } else if(part == 0) {
            pw = (double)d1[idx].real() * (double)d2[idx].real() + (double)d1[idx].imag() * (double)d2[idx].imag();
          } else {
            pw = (double)d1[idx].imag() * (double)d2[idx].real() - (double)d1[idx].real() * (double)d2[idx].imag();
          }

          if(method > 0) {
            const double wz = sinc_ratio(iz, nz);
            pw /= mas_window_sq(wx, wy, wz, method);
          }
          if(KPOW == 2) pw *= k2;

          k_thr[ib] += k;
          n_thr[ib] += 1.0;

          if(nells == 1 && ells[0] == 0) {
            p_thr[ib] += pw;
          } else {
            // mu keeps its sign. los = x or y reaches [-1, 1], los = z only the folded mu >= 0, which
            // even ell does not notice; odd ell of the real part vanishes for real fields anyway.
            const double klos = (los == 0 ? kx : (los == 1 ? ky : kz));
            const double mu = (k > 0.0) ? klos / k : 0.0;
            for(int ie = 0; ie < nells; ie++) {
              const double f = 2.0 * (double)ells[ie] + 1.0;
              p_thr[(size_t)ib + (size_t)ie * nbin] += f * std::legendre(ells[ie], mu) * pw;
            }
          }
        }
      }
    } // ix,iy,iz

#pragma omp critical
    {
      for(size_t i = 0; i < p_thr.size(); i++) sum_p[i] += p_thr[i];
      for(int i = 0; i < nbin; i++) sum_k[i] += k_thr[i];
      for(int i = 0; i < nbin; i++) nmodes[i] += n_thr[i];
    }
  } // omp parallel
}

// P(k) = (V / N^2) |delta(k)|^2 for FFTW's unnormalized forward transform, averaged over each bin.
// A bin with no mode is NaN rather than dropped.
static inline void finalize_power(const Grid3D &grid, const int nbin, const int nells, const double *sum_p,
                                  const double *sum_k, const double *nmodes, double *kout, double *pout)
{
  const double ncell = (double)grid.size();
  const double norm = grid.volume() / (ncell * ncell);

  for(int i = 0; i < nbin; i++) {
    if(nmodes[i] > 0.0) {
      kout[i] = sum_k[i] / nmodes[i];
      for(int ie = 0; ie < nells; ie++)
        pout[(size_t)i + (size_t)ie * nbin] = norm * sum_p[(size_t)i + (size_t)ie * nbin] / nmodes[i];
    } else {
      kout[i] = std::nan("");
      for(int ie = 0; ie < nells; ie++) pout[(size_t)i + (size_t)ie * nbin] = std::nan("");
    }
  }
}
