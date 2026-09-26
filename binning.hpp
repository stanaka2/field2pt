#pragma once

#include <cstdint>
#include <stdexcept>

// Bin edges supplied by the caller. Nothing is cut implicitly: a value outside [edges[0],
// edges[nbin]) simply falls in no bin, and the Nyquist region is kept unless the caller's edges
// leave it out.
struct BinSpec {
  const double *edges = nullptr;
  int nbin = 0;
  bool uniform = false; // equally spaced edges take the O(1) path
  double inv_width = 0.0;
};

static inline BinSpec make_bins(const double *edges, const int nbin)
{
  if(nbin <= 0) throw std::runtime_error("the number of bins must be positive");
  for(int i = 0; i < nbin; i++) {
    if(!(edges[i] < edges[i + 1])) throw std::runtime_error("bin edges must be strictly increasing");
  }

  BinSpec b;
  b.edges = edges;
  b.nbin = nbin;

  const double w = (edges[nbin] - edges[0]) / (double)nbin;
  bool uniform = true;
  for(int i = 0; i <= nbin; i++) {
    if(std::abs(edges[i] - (edges[0] + w * i)) > 1e-9 * w) {
      uniform = false;
      break;
    }
  }
  b.uniform = uniform;
  b.inv_width = (w > 0.0) ? 1.0 / w : 0.0;
  return b;
}

static inline int bin_index(const double x, const BinSpec &b)
{
  if(x < b.edges[0] || x >= b.edges[b.nbin]) return -1;

  if(b.uniform) {
    int i = (int)((x - b.edges[0]) * b.inv_width);
    if(i < 0) i = 0;
    if(i >= b.nbin) i = b.nbin - 1;
    // the multiply can land on the wrong side of an edge by a rounding step, so the guess is
    // checked against the edges themselves. The corrections almost never run.
    while(i > 0 && x < b.edges[i]) i--;
    while(i < b.nbin - 1 && x >= b.edges[i + 1]) i++;
    return i;
  }

  int lo = 0;
  int hi = b.nbin;
  while(hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if(x < b.edges[mid])
      hi = mid;
    else
      lo = mid;
  }
  return lo;
}
