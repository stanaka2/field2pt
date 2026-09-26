import numpy as np

from . import _binding


def method_type(method):
    # "NONE" is for a field not built by assignment, the one entry mass_assignment does not have
    method_map = {"NGP": 1, "CIC": 2, "TSC": 3, "PCS": 4, "NONE": 0}
    if isinstance(method, str):
        m = method.strip().upper()
        if m in method_map:
            return method_map[m]
    # np.integer as well as int, for an order read back from a file. bool is an int in Python,
    # which here would silently mean NGP
    elif isinstance(method, (int, np.integer)) and not isinstance(method, bool):
        if int(method) in method_map.values():
            return int(method)
    raise ValueError(
        "method must be one of 'NGP', 'CIC', 'TSC', 'PCS', 'none' or their corresponding integer codes 1,2,3,4,0"
    )


# a scalar means the same value on all three axes
def _normalize_nmesh(nmesh):
    if np.ndim(nmesh) == 0:
        n = int(nmesh)
        if n <= 0:
            raise ValueError("nmesh must be positive")
        return (n, n, n)
    if len(nmesh) != 3:
        raise ValueError("nmesh must be a scalar or length-3 sequence")
    out = tuple(int(x) for x in nmesh)
    if any(x <= 0 for x in out):
        raise ValueError("all nmesh elements must be positive")
    return out


def _normalize_lbox(lbox):
    if np.ndim(lbox) == 0:
        L = float(lbox)
        if L <= 0.0:
            raise ValueError("lbox must be positive")
        return (L, L, L)
    if len(lbox) != 3:
        raise ValueError("lbox must be a scalar or length-3 sequence")
    out = tuple(float(x) for x in lbox)
    if any(x <= 0.0 for x in out):
        raise ValueError("all lbox elements must be positive")
    return out


# A field is taken as it is, never converted: a conversion is a copy of the mesh, and nothing a
# deposit produces needs one. The binding would make the same copy silently, so the check stays.
def _check_field(field):
    if not isinstance(field, np.ndarray):
        raise TypeError(
            f"field must be a NumPy array, not {type(field).__name__}; a velocity or a moment "
            "comes as a tuple of components, so pass one component at a time"
        )
    if field.ndim != 3:
        raise ValueError(f"field must be a 3D array, not {field.ndim}D")
    if field.dtype not in (np.float32, np.float64):
        raise TypeError(
            f"field must be float32 or float64, not {field.dtype}; convert it with "
            ".astype(np.float32), the copy that would otherwise be made silently"
        )
    if not field.flags["C_CONTIGUOUS"]:
        raise ValueError(
            "field must be C-contiguous; np.ascontiguousarray(field) makes the copy explicitly"
        )
    return field


# The array cannot say which axis the field was distorted along, and a wrong guess is a plausible
# number (los="x" on a field distorted along z flips the quadrupole and halves it), so los is
# required for any non-zero multipole. Plane-parallel only: one axis serves the whole box.
def _normalize_los(los, ells):
    if los is None:
        if any(e > 0 for e in ells):
            raise ValueError(
                "los is required when a non-zero multipole is requested: pass the axis "
                "the field was distorted along, as 'x', 'y', 'z' or 0, 1, 2"
            )
        return 2  # unused: the monopole does not depend on the line of sight

    los_map = {"x": 0, "y": 1, "z": 2}
    if isinstance(los, str):
        l = los.strip().lower()
        if l in los_map:
            return los_map[l]
    elif isinstance(los, int):
        if los in (0, 1, 2):
            return los
    raise ValueError("los must be 'x', 'y', 'z' or 0, 1, 2")


# Re[d1 conj(d2)] is symmetric under k -> -k, so the power spectrum has no odd multipoles. The
# correlation function keeps the complex product, so a cross correlation can be asymmetric in s,
# and that odd part is what a gravitational redshift dipole is measured from.
# The backend takes any order; the limit is set here.
_ELL_MAX = 10


def _normalize_ells(ells, allow_odd=False, what="power spectrum"):
    if ells is None:
        return [0]
    if np.ndim(ells) == 0:
        ells = [ells]
    out = [int(x) for x in ells]
    if any(x < 0 or x > _ELL_MAX for x in out):
        raise ValueError(f"ells must be between 0 and {_ELL_MAX}")
    if not allow_odd and any(x % 2 for x in out):
        raise ValueError(
            f"odd multipoles vanish for the {what} of real fields, so only even ells "
            "are accepted here"
        )
    return out


# real accumulates Re[d1 conj(d2)], even ells only; imag accumulates Im[d1 conj(d2)], odd ells
# only, the Fourier-space view of the odd multipoles of a cross correlation.
def _normalize_part(part):
    if not isinstance(part, str) or part.strip().lower() not in ("real", "imag"):
        raise ValueError("part must be 'real' or 'imag'")
    return 0 if part.strip().lower() == "real" else 1


# P_shot = V / N, or V * sum(w^2) / sum(w)^2 for weighted particles, in the volume of this call so
# that the units cannot drift; a finished value through shotnoise puts the units on the caller.
# It is constant, so only the monopole takes it.
def _resolve_shotnoise(shotnoise, npart, weights, lbox):
    given = [x is not None for x in (shotnoise, npart, weights)]
    if sum(given) > 1:
        raise ValueError("pass at most one of shotnoise, npart and weights")

    volume = lbox[0] * lbox[1] * lbox[2]

    if npart is not None:
        n = float(npart)
        if n <= 0.0:
            raise ValueError("npart must be positive")
        return volume / n

    if weights is not None:
        w = np.asarray(weights)
        if w.ndim != 1:
            raise ValueError("weights must have shape (N,), one weight per particle")
        if w.size == 0:
            raise ValueError("weights must not be empty")
        # summed in float64 a chunk at a time, so that 1024^3 float32 weights are not copied whole
        total, total2 = 0.0, 0.0
        for i in range(0, w.size, 1 << 20):
            c = np.asarray(w[i : i + (1 << 20)], dtype=np.float64)
            total += float(c.sum())
            total2 += float(np.dot(c, c))
        if total <= 0.0:
            raise ValueError("the total weight must be positive")
        return volume * total2 / (total * total)

    return None if shotnoise is None else float(shotnoise)


def _subtract_shotnoise(power, ells, value):
    if value is None:
        return power
    if 0 in ells:
        power = power.copy()
        power[ells.index(0)] -= value
    return power


def _normalize_bin_range(bin_range):
    if np.ndim(bin_range) != 1 or len(bin_range) != 2:
        raise ValueError("range must be [lower, upper]")
    return float(bin_range[0]), float(bin_range[1])


# the default range of k: from the fundamental mode to the smallest Nyquist of the three axes
def _default_krange(shape, lbox):
    return min(2.0 * np.pi / L for L in lbox), min(
        np.pi * n / L for n, L in zip(shape, lbox)
    )


# the default range of s: one cell up to a quarter of the shortest box side, as far as a periodic
# box can be trusted before the correlation wraps onto itself
def _default_srange(shape, lbox):
    return min(L / n for n, L in zip(shape, lbox)), 0.25 * min(lbox)


# n bins between lo and hi, in equal steps of the quantity or of its log
def _count_edges(n, lo, hi, scale, what, shape, lbox):
    if hi <= lo:
        raise ValueError(
            f"no {what} range on a grid of shape {tuple(shape)} in a box {tuple(lbox)}: the upper end "
            f"{hi:.4g} is not above the lower {lo:.4g}; pass range, or bins as edges"
        )
    if scale == "log":
        if lo <= 0.0:
            raise ValueError("a log scale needs the lower end of range above 0")
        return np.geomspace(lo, hi, n + 1)
    return np.linspace(lo, hi, n + 1)


# scale lays out a bin count; explicit edges already say how they are spaced
def _normalize_scale(scale):
    if scale is None:
        return "linear"
    if not isinstance(scale, str) or scale.strip().lower() not in ("linear", "log"):
        raise ValueError("scale must be 'linear' or 'log'")
    return scale.strip().lower()


# bins is a count, laid out by range and scale, or the edges themselves, as numpy.histogram takes it
def _normalize_bins(bins, bin_range, scale, default_count):
    if bins is None or np.ndim(bins) == 0:
        n = default_count if bins is None else int(bins)
        if n <= 0:
            raise ValueError("bins must be a positive count or an array of edges")
        return n, _normalize_scale(scale), None
    if bin_range is not None or scale is not None:
        raise ValueError(
            "range and scale lay out a count; edges given as bins already say both"
        )
    e = np.asarray(bins, dtype=np.float64)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("bins must be a count or a 1D array of at least two edges")
    if np.any(np.diff(e) <= 0):
        raise ValueError("the edges in bins must be strictly increasing")
    return None, None, np.ascontiguousarray(e)


def _normalize_sedges(bins, bin_range, scale, shape, lbox):
    n, scale, e = _normalize_bins(bins, bin_range, scale, 32)
    if e is not None:
        return e
    if bin_range is None:
        lo, hi = _default_srange(shape, lbox)
    else:
        lo, hi = _normalize_bin_range(bin_range)
    return _count_edges(n, lo, hi, scale, "separation", shape, lbox)


def _normalize_kedges(bins, bin_range, scale, shape, lbox):
    n, scale, e = _normalize_bins(bins, bin_range, scale, 64)
    if e is not None:
        return e
    if bin_range is None:
        lo, hi = _default_krange(shape, lbox)
    else:
        lo, hi = _normalize_bin_range(bin_range)
    return _count_edges(n, lo, hi, scale, "k", shape, lbox)


# Reads the box and the assignment order off whatever object built the mesh, by duck typing: .lbox
# and .p are needed, .nmesh is checked against the array and .bc must be periodic when present.
def _from_grid(grid, shape):
    try:
        lbox = grid.lbox
        order = grid.p
    except AttributeError as exc:
        raise TypeError(
            "grid must expose .lbox and .p, as the object that built the mesh does"
        ) from exc

    nmesh = getattr(grid, "nmesh", None)
    if nmesh is not None and _normalize_nmesh(nmesh) != tuple(shape):
        raise ValueError(
            f"the field has shape {tuple(shape)} but the grid describes {_normalize_nmesh(nmesh)}"
        )

    bc = getattr(grid, "bc", None)
    if bc is not None:
        _check_periodic(bc)
    return lbox, order


# A Fourier transform treats every axis as periodic, so a mesh with an open axis is refused rather
# than measured with its edges touching. A single string stands for all three axes.
def _check_periodic(bc):
    axes = (bc, bc, bc) if isinstance(bc, str) else tuple(bc)
    if any(not isinstance(b, str) or b.strip().lower() != "periodic" for b in axes):
        raise ValueError(
            f"the grid has boundary {axes}, but the field is measured as periodic on every axis; "
            "a mesh with an open axis cannot be measured here"
        )


# lbox given alongside grid wins: the depositing package's lbox only scales the coordinates (1 for
# positions in [0,1)), while here it is the physical length that sets k and the power. The
# method has no second reading, so it cannot be overridden.
def _resolve_grid(grid, lbox, method, shape):
    if grid is not None:
        if method is not None:
            raise ValueError("method comes from grid and cannot be overridden")
        glbox, order = _from_grid(grid, shape)
        return (glbox if lbox is None else lbox), order
    if lbox is None or method is None:
        raise TypeError("lbox and method are required unless grid is given")
    return lbox, method


# nthreads left out follows the grid, as lbox and method do; 0 leaves the count to OpenMP
def _resolve_nthreads(nthreads, grid):
    if nthreads is not None:
        return int(nthreads)
    if grid is None:
        return 0
    return int(getattr(grid, "nthreads", 0))


# pk_ell2 or xi_ell2 is the quadrupole whatever else was requested, and a multipole that was not
# requested is not there, in dir() either
class _Multipoles:
    __slots__ = ("_values", "ells")

    def _ell(self, ell):
        if ell not in self.ells:
            raise AttributeError(f"multipole {ell} was not requested")
        return self._values[self.ells.index(ell)]

    def __getattr__(self, name):
        if name not in self._names:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return self._ell(self._names[name])

    def __dir__(self):
        return list(super().__dir__()) + [
            name for name, ell in self._names.items() if ell in self.ells
        ]


class PowerResult(_Multipoles):
    """k, the multipoles and the mode count of each bin.

    pk_ell0 through pk_ell10 reach the requested multipoles by their order, and pk the monopole.
    """

    __slots__ = ("k", "nmodes")
    _names = {f"pk_ell{ell}": ell for ell in range(_ELL_MAX + 1)}

    def __init__(self, k, power, nmodes, ells):
        self.k = k
        self._values = power
        self.nmodes = nmodes
        self.ells = tuple(ells)

    # the monopole, under the name a power spectrum usually goes by
    @property
    def pk(self):
        return self._ell(0)

    def __repr__(self):
        return f"PowerResult(nbin={len(self.k)}, ells={self.ells})"


class CorrelationResult(_Multipoles):
    """s, the multipoles and the cell count of each bin.

    xi_ell0 through xi_ell10 reach the requested multipoles by their order, and xi the monopole.
    """

    __slots__ = ("s", "ncells")
    _names = {f"xi_ell{ell}": ell for ell in range(_ELL_MAX + 1)}

    def __init__(self, s, correlation, ncells, ells):
        self.s = s
        self._values = correlation
        self.ncells = ncells
        self.ells = tuple(ells)

    # the monopole, under the name a correlation function usually goes by
    @property
    def xi(self):
        return self._ell(0)

    def __repr__(self):
        return f"CorrelationResult(nbin={len(self.s)}, ells={self.ells})"


class VelocityPowerResult:
    """P_thetatheta, P_omegaomega and, when delta was given, P_deltatheta, each a PowerResult."""

    __slots__ = ("theta", "omega", "cross")

    def __init__(self, theta, omega, cross):
        self.theta = theta
        self.omega = omega
        self.cross = cross

    def __repr__(self):
        return f"VelocityPowerResult(nbin={len(self.theta.k)}, ells={self.theta.ells}, cross={self.cross is not None})"


class Estimator:
    """Grid, settings and FFT workspace held across repeated measurements.

        est = f2.Estimator(grid=g)
        pk = est.power_spectrum(delta)
        xi = est.correlation_function(delta)

    grid is whatever built the mesh, and nthreads left out follows it too; nmesh, lbox and
    method can be given instead. The module level functions build a throwaway one on every call. One backend is kept per dtype, and the
    settings are read only, since the first measurement bakes them into the backend.
    """

    __slots__ = ("_nmesh", "_lbox", "_p", "_nthreads", "_backends")

    def __init__(self, nmesh=None, lbox=None, method=None, nthreads=None, grid=None):
        if grid is not None:
            if nmesh is not None or method is not None:
                raise ValueError(
                    "nmesh and method come from grid and cannot be overridden"
                )
            nmesh = getattr(grid, "nmesh", None)
            # Unlike the module functions, an Estimator exists before it is given a field, so the
            # shape has to come from the grid rather than being cross-checked against an array.
            if nmesh is None:
                raise TypeError(
                    "grid must expose .nmesh to build an Estimator, since there is no "
                    "field yet to take the shape from"
                )
            glbox, method = _from_grid(grid, _normalize_nmesh(nmesh))
            if lbox is None:
                lbox = glbox  # an explicit lbox wins; see _resolve_grid
        if nmesh is None or lbox is None or method is None:
            raise TypeError("nmesh, lbox and method are required unless grid is given")

        self._nmesh = _normalize_nmesh(nmesh)
        self._lbox = _normalize_lbox(lbox)
        self._p = method_type(method)
        self._nthreads = _resolve_nthreads(nthreads, grid)
        self._backends = {}

    @property
    def nmesh(self):
        return self._nmesh

    @property
    def lbox(self):
        return self._lbox

    @property
    def p(self):
        return self._p

    @property
    def method(self):
        return _METHOD_NAME.get(self._p, "none")

    @property
    def nthreads(self):
        return self._nthreads

    def _backend(self, field):
        cls = (
            _binding.Estimator32 if field.dtype == np.float32 else _binding.Estimator64
        )
        key = field.dtype.str
        if key not in self._backends:
            self._backends[key] = cls(self._nmesh, self._lbox, self._p, self._nthreads)
        return self._backends[key]

    @property
    def workspace_bytes(self):
        return sum(b.workspace_bytes for b in self._backends.values())

    def _check(self, field):
        f = _check_field(field)
        if f.shape != self.nmesh:
            raise ValueError(
                f"the field has shape {f.shape} but this estimator was built for {self.nmesh}"
            )
        return f

    # two fields are transformed in one precision, so they must already share it
    def _check_pair(self, field1, field2):
        f1 = self._check(field1)
        f2 = self._check(field2)
        if f2.dtype != f1.dtype:
            raise TypeError(
                f"the two fields must share a dtype, not {f1.dtype} and {f2.dtype}; convert one with "
                ".astype, the copy that would otherwise be made silently"
            )
        return f1, f2

    def power_spectrum(
        self,
        field,
        field2=None,
        bins=None,
        range=None,
        scale=None,
        ells=None,
        los=None,
        part="real",
        shotnoise=None,
        npart=None,
        weights=None,
    ):
        ipart = _normalize_part(part)
        if field2 is None:
            if ipart == 1:
                raise ValueError(
                    "the imaginary part of an auto spectrum is zero; part='imag' needs field2"
                )
            f1, f2_ = self._check(field), None
            ells = _normalize_ells(ells)
        else:
            if shotnoise is not None or npart is not None or weights is not None:
                raise ValueError(
                    "a cross spectrum of two tracers has no shot noise to subtract"
                )
            f1, f2_ = self._check_pair(field, field2)
            ells = _normalize_cross_ells(ells, ipart)
        kedges = _normalize_kedges(bins, range, scale, self.nmesh, self.lbox)
        los = _normalize_los(los, ells)
        sn = _resolve_shotnoise(shotnoise, npart, weights, self.lbox)
        k, p, n = self._backend(f1).power(f1, f2_, kedges, ipart, ells, los)
        return PowerResult(k, _subtract_shotnoise(p, ells, sn), n, ells)

    def interlaced_power_spectrum(
        self,
        field,
        field_shifted,
        bins=None,
        range=None,
        scale=None,
        ells=None,
        los=None,
        shotnoise=None,
        npart=None,
        weights=None,
    ):
        f1, f2_ = self._check_pair(field, field_shifted)
        kedges = _normalize_kedges(bins, range, scale, self.nmesh, self.lbox)
        ells = _normalize_ells(ells)
        los = _normalize_los(los, ells)
        sn = _resolve_shotnoise(shotnoise, npart, weights, self.lbox)
        # the pair is the "diag2" set of Interlaced, summed the same way; both fields already exist
        # here, which is the one mesh Interlaced saves by taking them one at a time
        b = self._backend(f1)
        pair = _SHIFT_SETS["diag2"]
        b.add_shifted(f1, pair[0], True)
        b.add_shifted(f2_, pair[1], False)
        k, p, n = b.power_of_sum(kedges, ells, los)
        return PowerResult(k, _subtract_shotnoise(p / 4.0, ells, sn), n, ells)

    # a velocity comes as three components, each a mesh of its own, all in one dtype
    def _check_vector(self, v):
        if not isinstance(v, (tuple, list)) or len(v) != 3:
            raise TypeError(
                "v must be the three components of a velocity, (vx, vy, vz)"
            )
        vx, vy, vz = (self._check(c) for c in v)
        if not vx.dtype == vy.dtype == vz.dtype:
            raise TypeError("the three components must share a dtype")
        return vx, vy, vz

    def velocity_power_spectrum(
        self,
        v,
        bins=None,
        range=None,
        scale=None,
        ells=None,
        los=None,
        delta=None,
    ):
        vx, vy, vz = self._check_vector(v)
        d = None
        if delta is not None:
            d = self._check(delta)
            if d.dtype != vx.dtype:
                raise TypeError(
                    f"delta must have the dtype of the velocity, {vx.dtype}, not {d.dtype}"
                )
        kedges = _normalize_kedges(bins, range, scale, self.nmesh, self.lbox)
        ells = _normalize_ells(ells)
        los = _normalize_los(los, ells)
        k, ptt, pww, pdt, n = self._backend(vx).velocity_power(
            vx, vy, vz, d, kedges, ells, los
        )
        cross = None if pdt is None else PowerResult(k, pdt, n, ells)
        return VelocityPowerResult(
            PowerResult(k, ptt, n, ells), PowerResult(k, pww, n, ells), cross
        )

    def correlation_function(
        self,
        field,
        field2=None,
        bins=None,
        range=None,
        scale=None,
        ells=None,
        los=None,
        drop_dc=True,
    ):
        if field2 is None:
            f1, f2_ = self._check(field), None
            ells = _normalize_ells(ells, allow_odd=False, what="auto correlation")
        else:
            f1, f2_ = self._check_pair(field, field2)
            ells = _normalize_ells(ells, allow_odd=True)
        sedges = _normalize_sedges(bins, range, scale, self.nmesh, self.lbox)
        los = _normalize_los(los, ells)
        s, x, n = self._backend(f1).correlation(
            f1, f2_, sedges, ells, los, bool(drop_dc)
        )
        return CorrelationResult(s, x, n, ells)

    def smooth(self, field, radius, kind="gaussian"):
        f = self._check(field)
        r = float(radius)
        if r <= 0.0:
            raise ValueError("radius must be positive")
        return self._backend(f).smooth(f, r, _smoothing_kind(kind))

    def __repr__(self):
        return (
            f"Estimator(nmesh={self.nmesh}, lbox={self.lbox}, method='{self.method}' (p={self.p}), "
            f"nthreads={self.nthreads})"
        )


def _cube(m):
    v = [j / float(m) for j in range(m)]
    return [(x, y, z) for x in v for y in v for z in v]


# Shift sets for Interlaced, in cell fractions. Image n survives only if exp(-2 pi i n.a) is 1 at
# every shift: half cells always keep (2,0,0), and diagonal shifts keep (1,-1,0).
_SHIFT_SETS = {
    "diag2": [(j / 2.0,) * 3 for j in range(2)],  # odd nx+ny+nz cancel
    "diag4": [(j / 4.0,) * 3 for j in range(4)],  # also nx+ny+nz = 2 mod 4
    "cube3": _cube(3),  # any component not a multiple of 3
    "cube4": _cube(4),  # any component not a multiple of 4
}


# No default: the number of fields is the whole cost of interlacing, and the set decides which
# images survive, so both are the caller's to state.
def _normalize_shifts(shifts):
    if isinstance(shifts, str):
        key = shifts.strip().lower()
        if key not in _SHIFT_SETS:
            raise ValueError(
                f"shifts must be one of {', '.join(_SHIFT_SETS)} or an (N,3) array of cell fractions"
            )
        return np.array(_SHIFT_SETS[key], dtype=np.float64)
    a = np.array(shifts, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape[0] < 1:
        raise ValueError(
            "shifts must be a set name or an (N,3) array of cell fractions"
        )
    if not np.all(np.isfinite(a)) or np.any(a < 0.0) or np.any(a >= 1.0):
        raise ValueError("every shift must lie in [0, 1) of a cell on each axis")
    if len(np.unique(a, axis=0)) != len(a):
        raise ValueError("shifts must not repeat")
    return a


class Interlaced:
    """Interlacing over a set of shifted deposits, taken one field at a time.

        it = f2.Interlaced("diag4", grid=g)
        for s in it.offsets():
            pos += s                  # moved in place, so the particles are not copied
            d = g.dens(pos)
            pos -= s
            d /= d.mean(); d -= 1.0
            it.add(d, s)              # transformed and summed here
            del d                     # before the next deposit makes its own
        res = it.power_spectrum(npart=len(pos))

    shifts is "diag2", "diag4", "cube3", "cube4" or an (N,3) array of cell fractions in [0, 1), and
    offsets() gives them as lengths, always positive. Memory does not grow with the number of
    shifts. The fields may come in any order, each once, all in the dtype of the first.
    """

    __slots__ = ("_est", "_shifts", "_added", "_dtype")

    def __init__(
        self, shifts, nmesh=None, lbox=None, method=None, nthreads=None, grid=None
    ):
        self._est = Estimator(
            nmesh=nmesh, lbox=lbox, method=method, nthreads=nthreads, grid=grid
        )
        self._shifts = _normalize_shifts(shifts)
        self._added = np.zeros(len(self._shifts), dtype=bool)
        self._dtype = None

    @property
    def nmesh(self):
        return self._est.nmesh

    @property
    def lbox(self):
        return self._est.lbox

    @property
    def p(self):
        return self._est.p

    @property
    def method(self):
        return self._est.method

    @property
    def nthreads(self):
        return self._est.nthreads

    @property
    def shifts(self):
        return self._shifts.copy()

    @property
    def workspace_bytes(self):
        return self._est.workspace_bytes

    def _cell(self):
        return np.array(self.lbox) / np.array(self.nmesh)

    def offsets(self, dtype=np.float64):
        """Return the shifts as lengths to add to the positions, one row per field."""
        return (self._shifts * self._cell()).astype(dtype)

    # which shift a field was built with, read back from the offset the caller moved the particles by
    def _index(self, shift):
        s = np.asarray(shift, dtype=np.float64)
        if s.shape != (3,):
            raise ValueError("shift must be one row of offsets()")
        hit = np.flatnonzero(
            np.all(np.abs(self._shifts - s / self._cell()) < 1e-6, axis=1)
        )
        if hit.size == 0:
            raise ValueError(f"shift {tuple(s)} is not one of offsets()")
        i = int(hit[0])
        if self._added[i]:
            raise ValueError(f"the field for shift {tuple(s)} has already been added")
        return i

    def add(self, field, shift):
        """Add the field deposited with the particles moved by shift, one row of offsets()."""
        i = self._index(shift)
        f = self._est._check(field)
        if self._dtype is None:
            self._dtype = f.dtype
        if f.dtype != self._dtype:
            raise TypeError(
                f"every field must have the dtype of the first, {self._dtype}, not {f.dtype}"
            )
        first = not self._added.any()
        self._est._backend(f).add_shifted(f, tuple(self._shifts[i]), first)
        self._added[i] = True

    def power_spectrum(
        self,
        bins=None,
        range=None,
        scale=None,
        ells=None,
        los=None,
        shotnoise=None,
        npart=None,
        weights=None,
    ):
        missing = int(np.count_nonzero(~self._added))
        if missing:
            raise ValueError(
                f"{missing} of the {len(self._shifts)} shifted fields have not been added"
            )
        kedges = _normalize_kedges(bins, range, scale, self.nmesh, self.lbox)
        ells = _normalize_ells(ells)
        los = _normalize_los(los, ells)
        sn = _resolve_shotnoise(shotnoise, npart, weights, self.lbox)
        k, p, n = self._est._backends[self._dtype.str].power_of_sum(kedges, ells, los)
        # the sum holds N fields rather than their mean
        p = p / float(len(self._shifts)) ** 2
        return PowerResult(k, _subtract_shotnoise(p, ells, sn), n, ells)

    def __repr__(self):
        return (
            f"Interlaced({len(self._shifts)} shifts, {int(self._added.sum())} added, nmesh={self.nmesh}, "
            f"lbox={self.lbox}, method='{self.method}' (p={self.p}))"
        )


_METHOD_NAME = {0: "none", 1: "NGP", 2: "CIC", 3: "TSC", 4: "PCS"}


def _smoothing_kind(kind):
    kinds = {"gaussian": 0, "tophat": 1, "top-hat": 1, "top_hat": 1}
    if not isinstance(kind, str) or kind.strip().lower() not in kinds:
        raise ValueError("kind must be 'gaussian' or 'tophat'")
    return kinds[kind.strip().lower()]


def _normalize_cross_ells(ells, ipart):
    if ipart == 0:
        return _normalize_ells(
            ells, allow_odd=False, what="real part of a cross spectrum"
        )
    out = _normalize_ells(ells, allow_odd=True)
    if any(e % 2 == 0 for e in out):
        raise ValueError(
            "the imaginary part of a cross spectrum has only odd multipoles, so pass "
            "odd ells"
        )
    return out


# The module level functions build a throwaway Estimator. That is the right shape for a single
# measurement; keep an Estimator instead when measuring the same grid more than once.
def _one_shot(field, lbox, method, grid, nthreads):
    f = _check_field(field)
    lbox, method = _resolve_grid(grid, lbox, method, f.shape)
    return f, Estimator(
        nmesh=f.shape,
        lbox=lbox,
        method=method,
        nthreads=_resolve_nthreads(nthreads, grid),
    )


# Smoothing applies a window; it never divides the assignment window out. So these take no
# method at all and read only the box off a grid. What that leaves folded into a small
# smoothing radius is in the README.
def _one_shot_window(field, lbox, grid, nthreads):
    f = _check_field(field)
    if grid is not None:
        glbox, _ = _from_grid(grid, f.shape)
        if lbox is None:
            lbox = glbox
    if lbox is None:
        raise TypeError("lbox is required unless grid is given")
    return f, Estimator(
        nmesh=f.shape, lbox=lbox, method=0, nthreads=_resolve_nthreads(nthreads, grid)
    )


def power_spectrum(
    field,
    field2=None,
    lbox=None,
    method=None,
    bins=None,
    range=None,
    scale=None,
    ells=None,
    los=None,
    part="real",
    shotnoise=None,
    npart=None,
    weights=None,
    grid=None,
    nthreads=None,
):
    """Compute the power spectrum of a gridded field, or the cross spectrum of two."""
    f, est = _one_shot(field, lbox, method, grid, nthreads)
    return est.power_spectrum(
        f,
        field2,
        bins=bins,
        range=range,
        scale=scale,
        ells=ells,
        los=los,
        part=part,
        shotnoise=shotnoise,
        npart=npart,
        weights=weights,
    )


def interlaced_power_spectrum(
    field,
    field_shifted,
    lbox=None,
    method=None,
    bins=None,
    range=None,
    scale=None,
    ells=None,
    los=None,
    shotnoise=None,
    npart=None,
    weights=None,
    grid=None,
    nthreads=None,
):
    """Compute the power spectrum of a field interlaced with its deposit moved by +half a cell."""
    f, est = _one_shot(field, lbox, method, grid, nthreads)
    return est.interlaced_power_spectrum(
        f,
        field_shifted,
        bins=bins,
        range=range,
        scale=scale,
        ells=ells,
        los=los,
        shotnoise=shotnoise,
        npart=npart,
        weights=weights,
    )


def velocity_power_spectrum(
    v,
    lbox=None,
    method=None,
    bins=None,
    range=None,
    scale=None,
    ells=None,
    los=None,
    delta=None,
    grid=None,
    nthreads=None,
):
    """Compute the power spectra of the divergence and the vorticity of a velocity field."""
    if not isinstance(v, (tuple, list)) or len(v) != 3:
        raise TypeError("v must be the three components of a velocity, (vx, vy, vz)")
    f, est = _one_shot(v[0], lbox, method, grid, nthreads)
    return est.velocity_power_spectrum(
        v,
        bins=bins,
        range=range,
        scale=scale,
        ells=ells,
        los=los,
        delta=delta,
    )


def correlation_function(
    field,
    field2=None,
    lbox=None,
    method=None,
    bins=None,
    range=None,
    scale=None,
    ells=None,
    los=None,
    drop_dc=True,
    grid=None,
    nthreads=None,
):
    """Compute the correlation function of a gridded field, or the cross correlation of two."""
    f, est = _one_shot(field, lbox, method, grid, nthreads)
    return est.correlation_function(
        f,
        field2,
        bins=bins,
        range=range,
        scale=scale,
        ells=ells,
        los=los,
        drop_dc=drop_dc,
    )


def smooth(field, lbox=None, radius=None, kind="gaussian", grid=None, nthreads=None):
    """Smooth a field with a Gaussian or top-hat window, leaving the assignment window in."""
    if radius is None:
        raise TypeError("radius is required")
    f, est = _one_shot_window(field, lbox, grid, nthreads)
    return est.smooth(f, radius, kind=kind)


def marked_field(
    field,
    lbox=None,
    radius=None,
    delta_s=None,
    power=1.0,
    kind="gaussian",
    grid=None,
    nthreads=None,
):
    """Weight a field by the mark of White (2016), built from its own smoothed version."""
    if radius is None:
        raise TypeError("radius is required")
    if delta_s is None:
        raise TypeError("delta_s is required")
    field = _check_field(field)
    ds = float(delta_s)
    pw = float(power)
    # worked in the buffer the smoothing hands back, so the mark costs no mesh beyond that one
    m = smooth(field, lbox=lbox, radius=radius, kind=kind, grid=grid, nthreads=nthreads)
    m += 1.0 + ds
    if m.min() <= 0.0:
        raise ValueError(
            "the mark is undefined where 1 + delta_s + delta_R <= 0: raise delta_s or "
            "the smoothing radius"
        )
    np.divide(1.0 + ds, m, out=m)
    m **= pw
    m *= field
    return m
