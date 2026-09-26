# field2pt

Power spectra, correlation functions and their multipoles of fields on a regular 3D grid, with the
numerical core in C++ and the FFT done by FFTW under OpenMP.

## Installation

```bash
pip install field2pt
```

Built from source with `-march=native`, together with mass_assignment. FFTW3 is required in single
and double precision with its OpenMP libraries:

```bash
apt install libfftw3-dev                               # Debian, Ubuntu
dnf install fftw-devel                                 # RHEL, Fedora
CMAKE_PREFIX_PATH=/path/to/fftw pip install field2pt   # FFTW elsewhere
```

## Usage

### Power spectrum

```python
import numpy as np
import field2pt as f2

res = f2.power_spectrum(delta, lbox=1000.0, method="TSC", bins=64)

res.k        # mean wavenumber of each bin
res.pk       # monopole, the same as res.pk_ell0
res.nmodes   # number of independent modes in each bin
```

`delta` is any field on a periodic grid. `lbox` sets the units of `k` and of the power. `method` is
the assignment scheme whose window is divided out, `"NGP"`, `"CIC"`, `"TSC"`, `"PCS"` or `"none"`,
and has no default: a wrong one skews the spectrum silently.

`Estimator` and `Interlaced` are made before they see a field, so they also take `nmesh`:

```python
est = f2.Estimator(nmesh=512, lbox=1000.0, method="TSC")
it = f2.Interlaced("diag4", nmesh=512, lbox=1000.0, method="TSC")
```

### With mass_assignment

```python
import mass_assignment as ma

g = ma.Grid(nmesh=512, lbox=1000.0, method="TSC")
delta = g.dens(pos)
delta /= delta.mean(); delta -= 1.0   # in place, no second mesh

res = f2.power_spectrum(delta, grid=g, bins=64)
```

`grid` gives `lbox`, `method` and `nthreads`, and is read by duck typing: `.lbox` and `.p` are
required, `.nmesh` is checked against the array, and every axis of `.bc` must be `"periodic"`.
The examples below take `grid=g`; `lbox` and `method` work in its place.

`lbox` passed with `grid` overrides it, for positions deposited in `[0, 1)` and measured in Mpc/h:

```python
g = ma.Grid(nmesh=512, lbox=1.0, method="TSC")
res = f2.power_spectrum(delta, grid=g, lbox=1000.0)
```

`method` cannot be overridden, and the settings of an `Estimator` are read-only.

### Multipoles

```python
res = f2.power_spectrum(delta, grid=g, ells=(0, 2, 4), los="z")
res.pk_ell0, res.pk_ell2, res.pk_ell4
```

- Orders up to 10. `res.pk_ell2` is the quadrupole whatever else `ells` holds; one not asked for
  does not exist.
- `los` is the axis of the distortion, required for any order above 0. A wrong axis gives a
  plausible wrong answer, not an error. Plane-parallel only; lightcones are not supported.
- An auto spectrum has only even orders. The odd orders of a cross spectrum come from `part="imag"`.
- On a cubic grid an isotropic field still shows orders from 4 up (2 is spared by symmetry): with
  bins one `k_f` wide, `ell = 4` reads 12 % of the monopole at `10 k_f` and `ell = 10` 4 % at
  `40 k_f`. Compare with a model averaged over the same modes.

### Repeated measurements

```python
est = f2.Estimator(grid=g)

pk = est.power_spectrum(delta)
xi = est.correlation_function(delta)
px = est.power_spectrum(delta1, delta2)

est.workspace_bytes   # what it is keeping alive
```

An `Estimator` keeps the FFT workspace across calls; the module functions make one per call. On
eight cores with float32, a power spectrum, a correlation function and a cross spectrum:

| nmesh | module functions | Estimator | workspace |
|---|---|---|---|
| 256 | 0.32 s | 0.19 s | 0.19 GB |
| 512 | 2.80 s | 1.84 s | 1.50 GB |

Pass the components of a velocity or a moment one at a time; a tuple is refused, since stacking
would copy them.

```python
pk = [est.power_spectrum(c) for c in g.central_moment(pos, vel, 2)]
```

### Shot noise

```python
res = f2.power_spectrum(delta, grid=g, npart=len(pos))
res = f2.power_spectrum(delta, grid=g, weights=w)
```

`npart` subtracts `V / N` and `weights` `V * sum(w^2) / sum(w)^2` from the monopole, in the units of
this call. `shotnoise` takes a finished value, whose units are then yours. Nothing is subtracted by
default, and these are the Poisson terms of a density contrast only. The correlation functions take
none: the term only reaches `s = 0`.

### Velocity and momentum

```python
w = m * vel[:, 0]            # one weight per particle
p = g.dens(pos, mass=w)      # momentum: sum(W m v) in each cell
p *= p.size / m.sum()        # (1 + delta) v, in units of velocity
res = f2.power_spectrum(p, grid=g, shotnoise=sn)

v = g.velc(pos, vel)[0]      # velocity: sum(W m v) / sum(W m) in each cell
res = f2.power_spectrum(v, grid=g)
```

- A momentum is a linear deposit, so the window correction is exact. Its Poisson term,
  `V * sum((m v)^2) / sum(m)^2`, goes through `shotnoise`.
- A velocity from `velc` is a ratio of deposits, so the window correction is approximate: 13 % low
  at `k_N / 2` for TSC with half a particle per cell, 85 % for NGP. No Poisson term applies. The
  same holds for `scalar`, `velc_norm` and the moments.
- Nothing checks which kind a field is. Subtract a bulk flow before a float32 transform.

### Divergence and vorticity

```python
res = f2.velocity_power_spectrum(g.velc(pos, vel), grid=g, delta=delta)

res.theta.pk   # spectrum of theta = div v
res.omega.pk   # spectrum of omega = curl v
res.cross.pk   # delta x theta, only when delta is given
```

`theta(k) = i k.v(k)` is exact on the mesh, where a fourth-order finite difference loses 28 % at
`k_N / 2`. The vorticity comes out of the same pass in two complex buffers, 1.08 GB at 512^3 in
float32. `theta` is `div v` in the input's units per unit of `lbox`; `-div v / (a H f)` flips the
sign of the cross spectrum. No shot noise, no interlacing.

`velc` is mass-weighted with empty cells at zero. For a known potential flow on uniform particles
with TSC, measured over true:

| particles per cell | `k_N / 4` | `k_N / 2` | `3 k_N / 4` | spurious omega / theta at `k_N / 2` |
|---|---|---|---|---|
| 8 | 0.999 | 0.999 | 1.045 | 0.007 |
| 1 | 0.985 | 0.999 | 1.551 | 0.075 |
| 0.5 | 0.974 | 1.018 | 2.271 | 0.178 |

A single component or the speed is an ordinary scalar field. The three component spectra add up to
`(P_theta + P_omega) / k^2`.

### Cross spectrum

```python
res = f2.power_spectrum(delta1, delta2, grid=g)
res = f2.power_spectrum(delta1, delta2, grid=g, ells=(1, 3), los="z", part="imag")
```

`part="real"` (the default) takes even orders, `part="imag"` odd ones; other combinations are
refused. No shot noise is subtracted from a cross spectrum.

### Binning

```python
f2.power_spectrum(delta, grid=g, bins=64)                                  # equal steps in k
f2.power_spectrum(delta, grid=g, bins=40, scale="log")                     # equal steps in log k
f2.power_spectrum(delta, grid=g, bins=40, scale="log", range=[0.01, 1.0])  # between set ends
```

`range` defaults to the fundamental mode up to the smallest Nyquist frequency. As in
`numpy.histogram`, `bins` also takes an array of edges, and then `range` and `scale` are refused. A
bin with no mode is NaN, and nothing outside the edges is used.

### Correlation function

```python
res = f2.correlation_function(delta, grid=g, bins=32)

res.s        # mean separation of each bin
res.xi       # monopole, the same as res.xi_ell0
res.ncells   # number of grid cells in each bin
```

The range defaults to one cell up to a quarter of the shortest side. `drop_dc=True` (the default)
removes the k = 0 mode.

```python
res = f2.correlation_function(delta1, delta2, grid=g, ells=(0, 1, 2), los="z")
res.xi_ell0, res.xi_ell1, res.xi_ell2
```

Separations keep their sign, so a cross correlation keeps its odd orders, such as the dipole of a
gravitational redshift in `xi_ell1`. Odd orders are refused for an auto correlation.

### Interlacing

```python
res = f2.interlaced_power_spectrum(delta, delta_shifted, grid=g)
```

The second field is deposited with every coordinate moved by `+ 0.5 * lbox / nmesh`. The direction
is not checked; the other way multiplies the power by `cos^2(k.H/2)`.

Any set of shifts, one field at a time:

```python
it = f2.Interlaced("diag4", grid=g)
for s in it.offsets():
    pos += s                  # in place, so the particles are not copied
    d = g.dens(pos)
    pos -= s
    d /= d.mean(); d -= 1.0
    it.add(d, s)
    del d                     # before the next deposit makes its own
res = it.power_spectrum(npart=len(pos))
```

Memory does not grow with the number of shifts, about 12 bytes a cell in float32 (1.6 GB at
512^3). The aliased share of white noise left with TSC in the shell 0.9 to 1.0 `k_N`:

| shifts | fields | residual |
|---|---|---|
| none | 1 | 1.4e-1 |
| `"diag2"`, half a cell along the diagonal | 2 | 4.2e-4 |
| `"diag4"`, quarter cells along the diagonal | 4 | 4.4e-5 |
| `"cube3"`, thirds of a cell on each axis | 27 | 1.9e-5 |
| `"cube4"`, quarters of a cell on each axis | 64 | 2.9e-6 |

An `(N,3)` array of cell fractions in `[0, 1)` gives any other set. For `pos + s` out of place, use
`it.offsets(pos.dtype)`, or float32 positions are promoted to float64.

### Smoothing and marked fields

```python
s = f2.smooth(delta, grid=g, radius=20.0, kind="gaussian")   # or "tophat"
m = f2.marked_field(delta, grid=g, radius=20.0, delta_s=4.0, power=1.0)
pk = f2.power_spectrum(m, grid=g)
```

The mark is that of White (2016), `m = ((1 + delta_s) / (1 + delta_s + delta_R)) ** power`, with
`delta_R` the field smoothed on `radius`. The assignment window is not divided out, which leaves
`delta_R` low by, for TSC:

| radius in cells | 10 | 5 | 2.6 | 2 | 1 |
|---|---|---|---|---|---|
| error | 0.1 % | 0.4 % | 1.5 % | 2.5 % | 8 % |

Both keep the dtype; `mesh_to_ptcl` takes float32.

### Non-cubic grids

```python
g = ma.Grid(nmesh=(512, 256, 128), lbox=(1000.0, 500.0, 250.0), method="TSC")
res = f2.power_spectrum(delta, grid=g)
```

### Threads

`nthreads` defaults to the grid's, else 0, which leaves it to OpenMP. A positive value sets the
OpenMP count of the whole process, mass_assignment included.

## Conventions

- The field is periodic.
- `P(k) = (V / N^2) |delta(k)|^2`, with V the box volume and N the number of cells.
- The power is divided by the square of the window `prod_i sinc(pi m_i / N_i)^p`. This removes the
  smoothing, not the aliased images.
- `nmodes` counts independent modes, so `sqrt(2 / nmodes)` is the relative error of a bin.
- Multipoles are normalized as `(2l+1) * int_0^1 f(l, mu) dmu`.

## Precision

float32 and float64 are accepted, the FFT follows the input and the binning is in double. Fields
are never converted, since that would copy the mesh: other dtypes, non-contiguous views and tuples
are refused, and two fields must share a dtype. float64 doubles the workspace. A large mean costs
float32 precision (6e-5 at a mean a thousand times the scatter), so subtract it in place first.

## Benchmarks

```bash
python benchmarks/bench.py --nmesh 256 512 --nthreads 8 --numpy
```

On eight cores with float32, agreeing with NumPy to 2e-7:

| nmesh | P(k) | xi(s) | interlaced | smooth | NumPy P(k) | peak RSS |
|---|---|---|---|---|---|---|
| 128 | 0.01 s | 0.01 s | 0.01 s | 0.00 s | 0.08 s | 0.17 GB |
| 256 | 0.04 s | 0.09 s | 0.10 s | 0.09 s | 0.78 s | 0.90 GB |
| 512 | 0.44 s | 1.17 s | 0.97 s | 0.88 s | - | 2.05 GB |

## License

Copyright (C) 2026 Satoshi Tanaka

field2pt is free software; you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version. It links against FFTW3, which is distributed under
the same license. The full text is in `LICENSE`.
