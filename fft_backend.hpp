#pragma once

#include <cstdint>
#include <complex>
#include <stdexcept>

#include <fftw3.h>

#include <omp.h>

#include "grid.hpp"

// FFTW keeps a separate set of symbols per precision, so the precision-dependent calls are
// gathered here and the rest of the package stays precision generic.
template <typename T>
struct fftw_api;

template <>
struct fftw_api<float> {
  using complex_t = fftwf_complex;
  using plan_t = fftwf_plan;

  static void init_threads()
  {
    fftwf_init_threads();
  }
  static void plan_with_nthreads(const int nt)
  {
    fftwf_plan_with_nthreads(nt);
  }
  static void *alloc_real(const int64_t n)
  {
    return fftwf_alloc_real(n);
  }
  static void *alloc_complex(const int64_t n)
  {
    return fftwf_alloc_complex(n);
  }
  static void free(void *p)
  {
    fftwf_free(p);
  }
  static void destroy(plan_t p)
  {
    fftwf_destroy_plan(p);
  }
  static void execute(plan_t p)
  {
    fftwf_execute(p);
  }

  static plan_t plan_r2c(const int nx, const int ny, const int nz, float *in, complex_t *out, const unsigned flags)
  {
    return fftwf_plan_dft_r2c_3d(nx, ny, nz, in, out, flags);
  }
  static plan_t plan_c2r(const int nx, const int ny, const int nz, complex_t *in, float *out, const unsigned flags)
  {
    return fftwf_plan_dft_c2r_3d(nx, ny, nz, in, out, flags);
  }
};

template <>
struct fftw_api<double> {
  using complex_t = fftw_complex;
  using plan_t = fftw_plan;

  static void init_threads()
  {
    fftw_init_threads();
  }
  static void plan_with_nthreads(const int nt)
  {
    fftw_plan_with_nthreads(nt);
  }
  static void *alloc_real(const int64_t n)
  {
    return fftw_alloc_real(n);
  }
  static void *alloc_complex(const int64_t n)
  {
    return fftw_alloc_complex(n);
  }
  static void free(void *p)
  {
    fftw_free(p);
  }
  static void destroy(plan_t p)
  {
    fftw_destroy_plan(p);
  }
  static void execute(plan_t p)
  {
    fftw_execute(p);
  }

  static plan_t plan_r2c(const int nx, const int ny, const int nz, double *in, complex_t *out, const unsigned flags)
  {
    return fftw_plan_dft_r2c_3d(nx, ny, nz, in, out, flags);
  }
  static plan_t plan_c2r(const int nx, const int ny, const int nz, complex_t *in, double *out, const unsigned flags)
  {
    return fftw_plan_dft_c2r_3d(nx, ny, nz, in, out, flags);
  }
};

// Sets the thread count once per process. FFTW keeps this in its own global state, so it is read
// back from the same public nthreads the rest of the package uses.
template <typename T>
static void fft_set_threads(const int nthreads)
{
  static bool started = false;
  if(!started) {
    fftw_api<T>::init_threads();
    started = true;
  }
  fftw_api<T>::plan_with_nthreads(nthreads > 0 ? nthreads : omp_get_max_threads());
}

// Owns an FFTW-aligned buffer. Used for the workspace the package allocates itself, never for the
// caller's arrays.
template <typename T>
class FFTBuffer {
public:
  FFTBuffer() = default;

  explicit FFTBuffer(const int64_t ncomplex) : n_(ncomplex)
  {
    ptr_ = static_cast<std::complex<T> *>(fftw_api<T>::alloc_complex(ncomplex));
    if(ptr_ == nullptr) throw std::runtime_error("failed to allocate the FFT workspace");
  }

  ~FFTBuffer()
  {
    if(ptr_ != nullptr) fftw_api<T>::free(ptr_);
  }

  FFTBuffer(const FFTBuffer &) = delete;
  FFTBuffer &operator=(const FFTBuffer &) = delete;

  FFTBuffer(FFTBuffer &&o) noexcept : ptr_(o.ptr_), n_(o.n_)
  {
    o.ptr_ = nullptr;
    o.n_ = 0;
  }

  FFTBuffer &operator=(FFTBuffer &&o) noexcept
  {
    if(this != &o) {
      if(ptr_ != nullptr) fftw_api<T>::free(ptr_);
      ptr_ = o.ptr_;
      n_ = o.n_;
      o.ptr_ = nullptr;
      o.n_ = 0;
    }
    return *this;
  }

  std::complex<T> *data() const
  {
    return ptr_;
  }
  int64_t size() const
  {
    return n_;
  }

private:
  std::complex<T> *ptr_ = nullptr;
  int64_t n_ = 0;
};

// Forward real-to-complex transform. The caller's real array is read only: FFTW guarantees that an
// out-of-place r2c plan leaves its input alone.
template <typename T>
static void fft_forward(const T *in, std::complex<T> *out, const Grid3D &grid, const int nthreads)
{
  fft_set_threads<T>(nthreads);

  using api = fftw_api<T>;
  typename api::plan_t plan =
      api::plan_r2c(grid.n[0], grid.n[1], grid.n[2], const_cast<T *>(in),
                    reinterpret_cast<typename api::complex_t *>(out), FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  if(plan == nullptr) throw std::runtime_error("failed to create the forward FFT plan");

  api::execute(plan);
  api::destroy(plan);
}

// Inverse complex-to-real transform. FFTW overwrites the complex input here, which is fine because
// it is always the package's own workspace.
template <typename T>
static void fft_backward(std::complex<T> *in, T *out, const Grid3D &grid, const int nthreads)
{
  fft_set_threads<T>(nthreads);

  using api = fftw_api<T>;
  typename api::plan_t plan = api::plan_c2r(grid.n[0], grid.n[1], grid.n[2],
                                            reinterpret_cast<typename api::complex_t *>(in), out, FFTW_ESTIMATE);
  if(plan == nullptr) throw std::runtime_error("failed to create the inverse FFT plan");

  api::execute(plan);
  api::destroy(plan);
}
