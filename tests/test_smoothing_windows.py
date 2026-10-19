import numpy as np
import pytest

import field2pt as f2
import mass_assignment as ma


@pytest.mark.parametrize("kind", ["gaussian", "tophat", "boxcar"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(9, 7, 5), (8, 6, 4)])
def test_analytic_fourier_mode_mean_dtype_and_input(kind, dtype, shape):
    box = np.array([2., 3., 5.])
    modes = np.array([1, -2, 1])
    coords = np.meshgrid(*(np.arange(n)/n for n in shape), indexing="ij")
    phase = sum(2*np.pi*m*x for m, x in zip(modes, coords))
    field = np.asarray(2.3 + np.cos(phase), dtype=dtype)
    original = field.copy()
    radius = .31
    q = 2*np.pi*modes/box*radius
    x = np.linalg.norm(q)
    if kind == "gaussian":
        amplitude = np.exp(-x*x/2)
    elif kind == "tophat":
        amplitude = 3*(np.sin(x)-x*np.cos(x))/x**3
    else:
        amplitude = np.prod(np.sinc(q/np.pi))
    est = f2.Estimator(nmesh=shape, lbox=box, method="TSC", nthreads=1)
    actual = est.smooth(field, radius, kind=kind)
    expected = 2.3 + amplitude*np.cos(phase)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(field, original)
    np.testing.assert_allclose(actual, expected, rtol=2e-6 if dtype == np.float32 else 2e-13)
    np.testing.assert_allclose(actual.mean(dtype=np.float64), field.mean(dtype=np.float64), rtol=4*np.finfo(dtype).eps if dtype == np.float32 else 2e-14)
    # Assignment deconvolution must not run during smoothing.
    other = f2.smooth(field, lbox=box, radius=radius, kind=kind, nthreads=1)
    np.testing.assert_allclose(actual, other, rtol=2e-6 if dtype == np.float32 else 2e-13)


@pytest.mark.parametrize("kind", ["gaussian", "boxcar"])
def test_velocity_filters_mass_and_momentum_before_dividing(kind):
    shape, box, radius = (15, 3, 5), (6., 2., 4.), .42
    coordinates = np.meshgrid(*(np.arange(n)*length/n for n, length in zip(shape, box)), indexing="ij")
    pos = np.column_stack([x.ravel() for x in coordinates])
    x = coordinates[0]*2*np.pi/box[0]
    mass = (2 + .6*np.cos(x)).ravel()
    vel = np.column_stack([(3 + 1.2*np.sin(x)).ravel(), np.full(x.size, -2.), np.zeros(x.size)])
    grid = ma.Grid(shape, box, "NGP", nthreads=1, dtype=np.float64)
    rho = grid.dens(pos, mass=mass)
    j = tuple(grid.dens(pos, mass=mass*vel[:, axis]) for axis in range(3))
    est = f2.Estimator(grid=grid, nthreads=1)
    density = est.smooth(rho, radius, kind=kind)
    vx = est.smooth(j[0], radius, kind=kind)/density
    q = 2*np.pi*radius/box[0]
    response = lambda n: np.exp(-.5*(n*q)**2) if kind == "gaussian" else np.sinc(n*q/np.pi)
    g1, g2 = response(1), response(2)
    expected = (6 + 1.8*g1*np.cos(x) + 2.4*g1*np.sin(x) + .36*g2*np.sin(2*x))/(2 + .6*g1*np.cos(x))
    np.testing.assert_allclose(vx, expected, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(est.smooth(j[1], radius, kind=kind)/density, -2., atol=3e-13)
    assert np.max(abs(vx-est.smooth(np.ascontiguousarray(vel[:, 0].reshape(shape)), radius, kind=kind))) > .001


def test_invalid_window_radius_and_open_boundary():
    field = np.ones((4, 4, 4))
    for radius in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="positive and finite"):
            f2.smooth(field, lbox=1., radius=radius)
    with pytest.raises(ValueError, match="kind"):
        f2.smooth(field, lbox=1., radius=.1, kind="invalid")
    with pytest.raises(ValueError, match="periodic"):
        f2.smooth(field, radius=.1, grid=ma.Grid(4, bc="open"))
