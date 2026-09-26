try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("field2pt")
except PackageNotFoundError:
    __version__ = "unknown"

from ._wrapper import (
    CorrelationResult,
    Estimator,
    Interlaced,
    PowerResult,
    VelocityPowerResult,
    correlation_function,
    interlaced_power_spectrum,
    marked_field,
    power_spectrum,
    smooth,
    velocity_power_spectrum,
)

__all__ = ["Estimator", "Interlaced"]
__all__ += ["power_spectrum", "interlaced_power_spectrum"]
__all__ += ["velocity_power_spectrum"]
__all__ += ["correlation_function"]
__all__ += ["smooth", "marked_field"]
__all__ += ["PowerResult", "CorrelationResult", "VelocityPowerResult"]
