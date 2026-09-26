#pragma once

#include <cmath>

#include "grid.hpp"

// Window of an assignment scheme of order p (1 NGP, 2 CIC, 3 TSC, 4 PCS), prod_i sinc(pi m_i / N_i)^p,
// taken per mode. Dividing by W^2 removes the smoothing but not the aliased images.
static inline double sinc_ratio(const int m, const int n)
{
  if(m == 0) return 1.0;
  const double x = M_PI * (double)m / (double)n;
  return std::sin(x) / x;
}

// W^2 for one mode, i.e. the factor the raw |delta(k)|^2 has to be divided by.
static inline double mas_window_sq(const double wx, const double wy, const double wz, const int p)
{
  double w = wx * wy * wz;
  double win = w;
  for(int q = 1; q < p; q++) win *= w;
  return win * win;
}
