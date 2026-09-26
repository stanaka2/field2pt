#pragma once

#include <cstdint>
#include <complex>
#include <vector>
#include <array>

#include "grid.hpp"
#include "binning.hpp"
#include "fft_backend.hpp"
#include "power.hpp"
#include "correlation.hpp"
#include "transform.hpp"

// Holds the grid, the settings and the FFT workspace across measurements: allocating and first
// touching a 539 MB complex buffer for 512^3 costs about a second, several times the transform.
// The second complex buffer and the real buffer are only allocated once something asks for them.
template <typename T>
class Estimator {
public:
  Estimator(const std::array<int, 3> &n, const std::array<double, 3> &lbox, const int method, const int nthreads)
      : grid_(make_grid(n, lbox)), method_(method), nthreads_(nthreads), c1_(grid_.csize())
  {}

  const Grid3D &grid() const
  {
    return grid_;
  }
  int method() const
  {
    return method_;
  }
  int nthreads() const
  {
    return nthreads_;
  }

  std::complex<T> *first()
  {
    return c1_.data();
  }

  std::complex<T> *second()
  {
    if(c2_.size() == 0) c2_ = FFTBuffer<T>(grid_.csize());
    return c2_.data();
  }

  T *real_buffer()
  {
    if((int64_t)rbuf_.size() != grid_.size()) rbuf_.resize(grid_.size());
    return rbuf_.data();
  }

  // bytes the workspace currently holds, so a caller can see what it is keeping alive
  int64_t workspace_bytes() const
  {
    int64_t b = c1_.size() * (int64_t)sizeof(std::complex<T>);
    b += c2_.size() * (int64_t)sizeof(std::complex<T>);
    b += (int64_t)rbuf_.size() * (int64_t)sizeof(T);
    return b;
  }

private:
  Grid3D grid_;
  int method_;
  int nthreads_;
  FFTBuffer<T> c1_;
  FFTBuffer<T> c2_;
  std::vector<T> rbuf_;
};
