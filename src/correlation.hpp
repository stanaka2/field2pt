#pragma once

#include <cmath>
#include <cstdint>
#include <complex>
#include <vector>

#include <omp.h>

#include "grid.hpp"
#include "binning.hpp"
#include "window.hpp"

// Replaces d1 by the spectral product, |d1|^2 or d1 conj(d2), whose imaginary part carries the odd
// multipoles. Every entry is visited: this rewrites the array, and a skipped entry would keep
// delta(k) and be read back as a power.
template <typename T>
static void spectral_product(std::complex<T> *d1, const std::complex<T> *d2, const Grid3D &grid, const int method,
                             const bool drop_dc, const int nthreads)
{
  const int nx = grid.n[0];
  const int ny = grid.n[1];
  const int nz = grid.n[2];
  const int nzc = grid.nz_c();

  if(nthreads > 0) omp_set_num_threads(nthreads);

#pragma omp parallel for collapse(2) schedule(static)
  for(int ix = 0; ix < nx; ix++) {
    for(int iy = 0; iy < ny; iy++) {

      const int mx = wave_index(ix, nx);
      const int my = wave_index(iy, ny);

      const double wx = sinc_ratio(mx, nx);
      const double wy = sinc_ratio(my, ny);

      const int64_t base = (int64_t)nzc * ((int64_t)iy + (int64_t)ny * (int64_t)ix);

      for(int iz = 0; iz < nzc; iz++) {

        const int64_t idx = base + iz;

        const double a_re = (double)d1[idx].real();
        const double a_im = (double)d1[idx].imag();

        double pr, pi;
        if(d2 == nullptr) {
          pr = a_re * a_re + a_im * a_im;
          pi = 0.0;
        } else {
          const double b_re = (double)d2[idx].real();
          const double b_im = (double)d2[idx].imag();
          pr = a_re * b_re + a_im * b_im;
          pi = a_im * b_re - a_re * b_im;
        }

        if(method > 0) {
          const double wz = sinc_ratio(iz, nz);
          const double w2 = mas_window_sq(wx, wy, wz, method);
          pr /= w2;
          pi /= w2;
        }

        if(drop_dc && mx == 0 && my == 0 && iz == 0) {
          pr = 0.0;
          pi = 0.0;
        }

        d1[idx] = std::complex<T>((T)pr, (T)pi);
      }
    }
  } // ix,iy,iz
}

// Bins the real-space correlation into separation bins and multipoles. Separations keep their sign,
// since folding them into the positive octant would lose the asymmetry that is the dipole.
template <typename T>
static void accumulate_correlation(const T *xi, const Grid3D &grid, const BinSpec &bins, const int nells,
                                   const int *ells, const int los, double *sum_xi, double *sum_s, double *ncell,
                                   const int nthreads)
{
  const int nx = grid.n[0];
  const int ny = grid.n[1];
  const int nz = grid.n[2];
  const int nbin = bins.nbin;

  const double dx = grid.lbox[0] / (double)nx;
  const double dy = grid.lbox[1] / (double)ny;
  const double dz = grid.lbox[2] / (double)nz;

  if(nthreads > 0) omp_set_num_threads(nthreads);

#pragma omp parallel
  {
    std::vector<double> x_thr((size_t)nbin * nells, 0.0);
    std::vector<double> s_thr(nbin, 0.0);
    std::vector<double> n_thr(nbin, 0.0);

#pragma omp for collapse(2) schedule(static)
    for(int ix = 0; ix < nx; ix++) {
      for(int iy = 0; iy < ny; iy++) {

        const double sx = (double)wave_index(ix, nx) * dx;
        const double sy = (double)wave_index(iy, ny) * dy;
        const double sxy2 = sx * sx + sy * sy;

        const int64_t base = (int64_t)nz * ((int64_t)iy + (int64_t)ny * (int64_t)ix);

        for(int iz = 0; iz < nz; iz++) {

          const double sz = (double)wave_index(iz, nz) * dz;
          const double s = std::sqrt(sxy2 + sz * sz);

          const int ib = bin_index(s, bins);
          if(ib < 0) continue;

          const double v = (double)xi[base + iz];

          s_thr[ib] += s;
          n_thr[ib] += 1.0;

          if(nells == 1 && ells[0] == 0) {
            x_thr[ib] += v;
          } else {
            const double slos = (los == 0 ? sx : (los == 1 ? sy : sz));
            const double mu = (s > 0.0) ? slos / s : 0.0;
            for(int ie = 0; ie < nells; ie++) {
              const double f = 2.0 * (double)ells[ie] + 1.0;
              x_thr[(size_t)ib + (size_t)ie * nbin] += f * std::legendre(ells[ie], mu) * v;
            }
          }
        }
      }
    } // ix,iy,iz

#pragma omp critical
    {
      for(size_t i = 0; i < x_thr.size(); i++) sum_xi[i] += x_thr[i];
      for(int i = 0; i < nbin; i++) sum_s[i] += s_thr[i];
      for(int i = 0; i < nbin; i++) ncell[i] += n_thr[i];
    }
  } // omp parallel
}

// FFTW is unnormalized both ways, so the round trip leaves N^2 behind. Over the full grid a bin
// average of L_ell-weighted values is (2l+1)/2 times the integral over mu in [-1, 1].
static inline void finalize_correlation(const Grid3D &grid, const int nbin, const int nells, const double *sum_xi,
                                        const double *sum_s, const double *ncell, double *sout, double *xout)
{
  const double n2 = (double)grid.size() * (double)grid.size();

  for(int i = 0; i < nbin; i++) {
    if(ncell[i] > 0.0) {
      sout[i] = sum_s[i] / ncell[i];
      for(int ie = 0; ie < nells; ie++)
        xout[(size_t)i + (size_t)ie * nbin] = sum_xi[(size_t)i + (size_t)ie * nbin] / (ncell[i] * n2);
    } else {
      sout[i] = std::nan("");
      for(int ie = 0; ie < nells; ie++) xout[(size_t)i + (size_t)ie * nbin] = std::nan("");
    }
  }
}
