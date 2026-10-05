#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <complex>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "grid.hpp"
#include "binning.hpp"
#include "window.hpp"
#include "fft_backend.hpp"
#include "power.hpp"
#include "correlation.hpp"
#include "transform.hpp"
#include "estimator.hpp"

namespace py = pybind11;

template <typename T>
static void check_shape(const Estimator<T> &est, const py::array_t<T, py::array::c_style> &f, const char *what)
{
  if(f.ndim() != 3) throw std::runtime_error(std::string(what) + " must be a 3D array");
  const Grid3D &g = est.grid();
  if(f.shape(0) != g.n[0] || f.shape(1) != g.n[1] || f.shape(2) != g.n[2])
    throw std::runtime_error(std::string(what) + " does not have the shape this estimator was built for");
}

// The binned sums as (x, value, count) arrays, x being k or s. finalize is finalize_power or
// finalize_correlation, which differ only in the normalization.
using finalize_fn = void (*)(const Grid3D &, const int, const int, const double *, const double *, const double *,
                             double *, double *);

static py::tuple pack_bins(finalize_fn finalize, const Grid3D &grid, const int nbin, const int nells,
                           const std::vector<double> &sum_v, const std::vector<double> &sum_x,
                           const std::vector<double> &count)
{
  py::array_t<double> xout({nbin}), vout({nells, nbin}), nout({nbin});
  finalize(grid, nbin, nells, sum_v.data(), sum_x.data(), count.data(), static_cast<double *>(xout.request().ptr),
           static_cast<double *>(vout.request().ptr));
  std::copy(count.begin(), count.end(), static_cast<double *>(nout.request().ptr));
  return py::make_tuple(xout, vout, nout);
}

// --- 1. Power spectrum ---

template <typename T>
static py::tuple power_impl(Estimator<T> &est, py::array_t<T, py::array::c_style> f1, py::object f2obj,
                            py::array_t<double> kedges, const int part, std::vector<int> ells, const int los)
{
  check_shape(est, f1, "field");
  const Grid3D &grid = est.grid();
  const int nbin = (int)kedges.size() - 1;
  const BinSpec bins = make_bins(kedges.data(), nbin);
  const int nells = (int)ells.size();

  py::array_t<T, py::array::c_style> holder;
  const bool cross = !f2obj.is_none();
  if(cross) {
    holder = py::cast<py::array_t<T, py::array::c_style>>(f2obj);
    check_shape(est, holder, "the second field");
  }

  std::vector<double> sum_p((size_t)nbin * nells, 0.0), sum_k(nbin, 0.0), nmodes(nbin, 0.0);
  {
    py::gil_scoped_release release;
    fft_forward<T>(f1.data(), est.first(), grid, est.nthreads());
    if(cross) fft_forward<T>(holder.data(), est.second(), grid, est.nthreads());
    accumulate_power<0, T>(est.first(), cross ? est.second() : nullptr, grid, est.method(), part, bins, nells,
                           ells.data(), los, sum_p.data(), sum_k.data(), nmodes.data(), est.nthreads());
  } // gil release
  return pack_bins(finalize_power, grid, nbin, nells, sum_p, sum_k, nmodes);
}

// --- 1.1 Interlacing ---

// One field at a time: the first buffer holds the sum and the second the field being added, so
// memory does not grow with the number of shifts. An unshifted first field goes straight in.
template <typename T>
static void add_shifted_impl(Estimator<T> &est, py::array_t<T, py::array::c_style> f,
                             const std::array<double, 3> &shift, const bool first)
{
  check_shape(est, f, "field");
  const Grid3D &grid = est.grid();
  const bool moved = (shift[0] != 0.0 || shift[1] != 0.0 || shift[2] != 0.0);
  {
    py::gil_scoped_release release;
    if(first) {
      fft_forward<T>(f.data(), est.first(), grid, est.nthreads());
      if(moved) interlace_add<T>(est.first(), nullptr, grid, shift.data(), est.nthreads());
    } else {
      fft_forward<T>(f.data(), est.second(), grid, est.nthreads());
      interlace_add<T>(est.first(), est.second(), grid, shift.data(), est.nthreads());
    }
  } // gil release
}

template <typename T>
static py::tuple power_of_sum_impl(Estimator<T> &est, py::array_t<double> kedges, std::vector<int> ells, const int los)
{
  const Grid3D &grid = est.grid();
  const int nbin = (int)kedges.size() - 1;
  const BinSpec bins = make_bins(kedges.data(), nbin);
  const int nells = (int)ells.size();

  std::vector<double> sum_p((size_t)nbin * nells, 0.0), sum_k(nbin, 0.0), nmodes(nbin, 0.0);
  {
    py::gil_scoped_release release;
    accumulate_power<0, T>(est.first(), nullptr, grid, est.method(), 0, bins, nells, ells.data(), los, sum_p.data(),
                           sum_k.data(), nmodes.data(), est.nthreads());
  } // gil release
  return pack_bins(finalize_power, grid, nbin, nells, sum_p, sum_k, nmodes);
}

// --- 1.2 Divergence and vorticity ---

// theta = div v is summed into the first buffer while each component passes through the second, and
// k^2 |v|^2 is binned on the way, so |k x v|^2 = k^2 |v|^2 - |theta|^2 needs no third buffer.
template <typename T>
static py::tuple velocity_power_impl(Estimator<T> &est, py::array_t<T, py::array::c_style> vx,
                                     py::array_t<T, py::array::c_style> vy, py::array_t<T, py::array::c_style> vz,
                                     py::object dobj, py::array_t<double> kedges, std::vector<int> ells, const int los)
{
  check_shape(est, vx, "vx");
  check_shape(est, vy, "vy");
  check_shape(est, vz, "vz");
  const Grid3D &grid = est.grid();
  const int nbin = (int)kedges.size() - 1;
  const BinSpec bins = make_bins(kedges.data(), nbin);
  const int nells = (int)ells.size();

  py::array_t<T, py::array::c_style> holder;
  const bool cross = !dobj.is_none();
  if(cross) {
    holder = py::cast<py::array_t<T, py::array::c_style>>(dobj);
    check_shape(est, holder, "delta");
  }

  const T *comp[3] = {vx.data(), vy.data(), vz.data()};
  std::vector<double> sum_vv((size_t)nbin * nells, 0.0), sum_tt((size_t)nbin * nells, 0.0);
  std::vector<double> sum_dt((size_t)nbin * nells, 0.0);
  std::vector<double> sum_k(nbin, 0.0), nmodes(nbin, 0.0), k_again(nbin, 0.0), n_again(nbin, 0.0);
  {
    py::gil_scoped_release release;
    for(int a = 0; a < 3; a++) {
      fft_forward<T>(comp[a], est.second(), grid, est.nthreads());
      accumulate_power<2, T>(est.second(), nullptr, grid, est.method(), 0, bins, nells, ells.data(), los, sum_vv.data(),
                             k_again.data(), n_again.data(), est.nthreads());
      divergence_add<T>(est.first(), est.second(), grid, a, a == 0, est.nthreads());
    }
    accumulate_power<0, T>(est.first(), nullptr, grid, est.method(), 0, bins, nells, ells.data(), los, sum_tt.data(),
                           sum_k.data(), nmodes.data(), est.nthreads());
    if(cross) {
      fft_forward<T>(holder.data(), est.second(), grid, est.nthreads());
      accumulate_power<0, T>(est.second(), est.first(), grid, est.method(), 0, bins, nells, ells.data(), los,
                             sum_dt.data(), k_again.data(), n_again.data(), est.nthreads());
    }
  } // gil release

  std::vector<double> sum_ww(sum_vv.size());
  for(size_t i = 0; i < sum_ww.size(); i++) sum_ww[i] = sum_vv[i] - sum_tt[i];

  py::array_t<double> kout({nbin}), ptt({nells, nbin}), pww({nells, nbin}), pdt({nells, nbin}), nout({nbin});
  double *kp = static_cast<double *>(kout.request().ptr);
  finalize_power(grid, nbin, nells, sum_tt.data(), sum_k.data(), nmodes.data(), kp,
                 static_cast<double *>(ptt.request().ptr));
  finalize_power(grid, nbin, nells, sum_ww.data(), sum_k.data(), nmodes.data(), kp,
                 static_cast<double *>(pww.request().ptr));
  finalize_power(grid, nbin, nells, sum_dt.data(), sum_k.data(), nmodes.data(), kp,
                 static_cast<double *>(pdt.request().ptr));
  std::copy(nmodes.begin(), nmodes.end(), static_cast<double *>(nout.request().ptr));
  return py::make_tuple(kout, ptt, pww, cross ? py::object(pdt) : py::object(py::none()), nout);
}

// --- 2. Correlation function ---

template <typename T>
static py::tuple correlation_impl(Estimator<T> &est, py::array_t<T, py::array::c_style> f1, py::object f2obj,
                                  py::array_t<double> sedges, std::vector<int> ells, const int los, const bool drop_dc)
{
  check_shape(est, f1, "field");
  const Grid3D &grid = est.grid();
  const int nbin = (int)sedges.size() - 1;
  const BinSpec bins = make_bins(sedges.data(), nbin);
  const int nells = (int)ells.size();

  py::array_t<T, py::array::c_style> holder;
  const bool cross = !f2obj.is_none();
  if(cross) {
    holder = py::cast<py::array_t<T, py::array::c_style>>(f2obj);
    check_shape(est, holder, "the second field");
  }

  std::vector<double> sum_xi((size_t)nbin * nells, 0.0), sum_s(nbin, 0.0), ncell(nbin, 0.0);
  {
    py::gil_scoped_release release;
    fft_forward<T>(f1.data(), est.first(), grid, est.nthreads());
    if(cross) fft_forward<T>(holder.data(), est.second(), grid, est.nthreads());
    spectral_product<T>(est.first(), cross ? est.second() : nullptr, grid, est.method(), drop_dc, est.nthreads());
    T *xi = est.real_buffer();
    fft_backward<T>(est.first(), xi, grid, est.nthreads());
    accumulate_correlation<T>(xi, grid, bins, nells, ells.data(), los, sum_xi.data(), sum_s.data(), ncell.data(),
                              est.nthreads());
  } // gil release
  return pack_bins(finalize_correlation, grid, nbin, nells, sum_xi, sum_s, ncell);
}

// --- 3. Smoothing ---

template <typename T>
static py::array_t<T> smooth_impl(Estimator<T> &est, py::array_t<T, py::array::c_style> f, const double radius,
                                  const int kind)
{
  check_shape(est, f, "field");
  const Grid3D &grid = est.grid();
  if(!std::isfinite(radius) || radius <= 0.0)
    throw std::runtime_error("the smoothing radius must be positive and finite");

  py::array_t<T> out({grid.n[0], grid.n[1], grid.n[2]});
  T *optr = static_cast<T *>(out.request().ptr);
  {
    py::gil_scoped_release release;
    fft_forward<T>(f.data(), est.first(), grid, est.nthreads());
    smooth_spectrum<T>(est.first(), grid, radius, kind, est.nthreads());
    fft_backward<T>(est.first(), optr, grid, est.nthreads());

    const double inv = 1.0 / (double)grid.size();
    const int64_t ncell = grid.size();
#pragma omp parallel for schedule(static)
    for(int64_t i = 0; i < ncell; i++) optr[i] = (T)((double)optr[i] * inv);
  } // gil release
  return out;
}

// --- Python Bindings ---

template <typename T>
static void bind_dtype(py::module_ &m, const char *suffix)
{
  using Est = Estimator<T>;

  py::class_<Est>(m, (std::string("Estimator") + suffix).c_str())
      .def(py::init<const std::array<int, 3> &, const std::array<double, 3> &, int, int>(), py::arg("nmesh"),
           py::arg("lbox"), py::arg("method"), py::arg("nthreads"))
      .def_property_readonly("workspace_bytes", &Est::workspace_bytes)
      .def("power", &power_impl<T>, py::arg("field"), py::arg("field2"), py::arg("kedges"), py::arg("part"),
           py::arg("ells"), py::arg("los"))
      .def("add_shifted", &add_shifted_impl<T>, py::arg("field"), py::arg("shift"), py::arg("first"))
      .def("power_of_sum", &power_of_sum_impl<T>, py::arg("kedges"), py::arg("ells"), py::arg("los"))
      .def("velocity_power", &velocity_power_impl<T>, py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("delta"),
           py::arg("kedges"), py::arg("ells"), py::arg("los"))
      .def("correlation", &correlation_impl<T>, py::arg("field"), py::arg("field2"), py::arg("sedges"), py::arg("ells"),
           py::arg("los"), py::arg("drop_dc"))
      .def("smooth", &smooth_impl<T>, py::arg("field"), py::arg("radius"), py::arg("kind"));
}

PYBIND11_MODULE(_binding, m)
{
  m.doc() = "Two-point statistics of gridded fields (FFTW + OpenMP)";
  bind_dtype<float>(m, "32");
  bind_dtype<double>(m, "64");
}
