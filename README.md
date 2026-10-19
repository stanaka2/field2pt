# field2pt

Power spectra, correlation functions, and multipoles of fields on regular 3D
grids, using FFTW and OpenMP.

## Installation

```bash
pip install field2pt
```

FFTW3 is required in single and double precision with its OpenMP libraries:

```bash
apt install libfftw3-dev                               # Debian, Ubuntu
dnf install fftw-devel                                 # RHEL, Fedora
CMAKE_PREFIX_PATH=/path/to/fftw pip install field2pt   # FFTW elsewhere
```

## Inputs

Fields are caller-provided C-contiguous NumPy arrays with shape `(nx, ny, nz)`
and dtype float32 or float64. They are treated as periodic. `lbox` is a scalar
or three physical box lengths.

`method` describes the assignment window to remove: `"NGP"`, `"CIC"`,
`"TSC"`, `"PCS"`, or `"none"`. Use `"none"` when no such assignment window
was applied.

## Usage

```python
import numpy as np
import field2pt as f2

rng = np.random.default_rng(42)
delta = rng.normal(size=(128, 128, 128)).astype(np.float32)

est = f2.Estimator(nmesh=delta.shape, lbox=1000.0, method="none")

pk = est.power_spectrum(delta, bins=64)
xi = est.correlation_function(delta, bins=32)

pk.k, pk.pk, pk.nmodes
xi.s, xi.xi, xi.ncells
```

An `Estimator` retains FFT workspaces for repeated measurements. Module-level
functions provide the same operations for one-off calls.

## Multipoles and cross statistics

```python
pk = est.power_spectrum(delta, ells=(0, 2, 4), los="z")
pk.pk_ell0, pk.pk_ell2, pk.pk_ell4

cross = est.power_spectrum(delta1, delta2)
odd = est.power_spectrum(delta1, delta2, ells=(1, 3), los="z", part="imag")

xi = est.correlation_function(delta1, delta2, ells=(0, 1, 2), los="z")
```

Orders up to 10 are supported. `los` is required for nonzero multipoles.
Auto spectra and the real part of cross spectra use even orders; the imaginary
part of a cross spectrum uses odd orders.

`bins` may be a positive count or an array of edges. Counted bins accept
`range=[lower, upper]` and `scale="linear"` or `"log"`. Empty bins contain
NaN.

## Shot noise

```python
pk = est.power_spectrum(delta, npart=number_of_particles)
pk = est.power_spectrum(delta, weights=particle_weights)
pk = est.power_spectrum(delta, shotnoise=value)
```

These subtract a constant Poisson term from the monopole. Nothing is
subtracted by default or from cross spectra.

## Velocity fields

```python
velocity = (velocity_x, velocity_y, velocity_z)
result = est.velocity_power_spectrum(velocity, delta=delta)

result.theta.pk   # divergence spectrum
result.omega.pk   # vorticity spectrum
result.cross.pk   # delta x divergence, when delta is supplied
```

The components must have the same shape and dtype as `delta`.

## Interlacing

For the common half-cell pair:

```python
pk = est.interlaced_power_spectrum(delta, delta_shifted)
```

For more shifts, add one caller-prepared field at a time:

```python
it = f2.Interlaced("diag4", nmesh=delta.shape, lbox=1000.0, method="none")
for shifted_field, shift in zip(shifted_fields, it.offsets()):
    it.add(shifted_field, shift)
pk = it.power_spectrum()
```

Built-in sets are `"diag2"`, `"diag4"`, `"cube3"`, and `"cube4"`.

## Smoothing and marked fields

```python
smoothed = est.smooth(delta, radius=20.0, kind="gaussian")
marked = f2.marked_field(
    delta, lbox=1000.0, radius=20.0, delta_s=4.0, power=1.0
)
```

Smoothing supports three windows and preserves the input dtype. `radius` is
in the same physical units as `lbox`:

| `kind` | Meaning of `radius` | Fourier window |
| --- | --- | --- |
| `"gaussian"` (default) | Standard deviation R | exp(-k²R²/2) |
| `"tophat"` | Spherical radius R | 3[sin(kR)-kR cos(kR)]/(kR)³ |
| `"boxcar"` | Half-width R; full width 2R along each axis | Product of sin(kᵢR)/(kᵢR) |

The boxcar is an axis-aligned cube, distinct from the spherical top-hat. These
are continuous Fourier windows, not integer-cell moving averages. Smoothing
preserves the DC mode and leaves the assignment window in place. All fields
are periodic; no edge padding or open-boundary filtering is performed.

For density-weighted velocity, filter mass and momentum with the same window
before division. These fields may come from saved meshes or particle assignment;
no new `mass_assignment` API is required. Analytic backgrounds belong to the
caller and must be included before division. An additional filter on the
resulting velocity is a distinct operation. Smoothing and a power measurement
with `method="none"` retain the assignment window; no shot noise is subtracted
by default.

## Grid metadata

Instead of passing `nmesh`, `lbox`, and `method` separately, `grid=` accepts an
object exposing `.lbox` and `.p`. Optional `.nmesh` and `.bc` attributes are
validated when present. Every boundary must be periodic.

`nthreads=0` uses the OpenMP default. A positive value sets the estimator's
thread count.

## Conventions

- `P(k) = (V/N^2) |delta(k)|^2`, where `N` is the number of grid cells.
- Assignment-window deconvolution removes smoothing but not aliased images.
- `nmodes` counts independent Fourier modes.
- float32 and float64 inputs retain their precision; arrays are not converted.

## License

GPL-2.0-or-later; see `LICENSE`.
