"""NASA DAS 3.2.7 orbital-lifetime reference (validation gate 5), via its Fortran.dll.

DAS 3.2.7 ships a ctypes-callable Fortran.dll. LIFEDWELL propagates with PROP3D
(J2-J4, luni-solar, SRP with shadow, Jacchia-77 drag in an oblate rotating
atmosphere, Cd = 2.2, NASA ODPO F10.7 table + curve-fit forecast). Windows only;
the DLL reads data\\solarflux.bin relative to the working directory, so calls
run with the install directory as cwd.

DAS is NASA software under a Software Usage Agreement; this module only calls a
local install and ships none of it.
"""

from __future__ import annotations

import ctypes as ct
import os
from contextlib import contextmanager
from pathlib import Path

_LIB = None


def das_home() -> Path:
    env = os.environ.get("TALOS_DAS_HOME")
    return Path(env) if env else Path(os.environ.get("LOCALAPPDATA", "")) / "NASA" / "DAS3.2.7"


@contextmanager
def _in_dir(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _lib():
    global _LIB
    if _LIB is None:
        home = das_home()
        dll = home / "Fortran.dll"
        if not dll.exists():
            raise FileNotFoundError(f"DAS Fortran.dll not found at {dll}; set TALOS_DAS_HOME")
        os.add_dll_directory(str(home))
        with _in_dir(home):
            lib = ct.CDLL(str(dll))
        d, i = ct.POINTER(ct.c_double), ct.POINTER(ct.c_int)
        lib.LIFEDWELL.argtypes = [d] * 10 + [i, i]
        lib.LIFEDWELL.restype = None
        _LIB = lib
    return _LIB


# IMSG codes from the NASA example wrapper
IMSG = {0: "no message", 1: "stays above LEO >100 yr (not processed)", 2: "reentered",
        3: "did not reenter before end year"}


def lifetime(start_year: float, perigee_km: float, apogee_km: float, inc_deg: float,
             area_to_mass: float, raan_deg: float = 0.0, argp_deg: float = 0.0,
             end_year: float = 2130.0) -> tuple[float, float, int]:
    """DAS LIFEDWELL: (lifetime [yr] from start_year, LEO dwell time [yr], message code).

    ``area_to_mass`` is average cross-section / mass [m^2/kg] (TALOS: 1 / areal_density).
    """
    lib = _lib()
    args = [ct.c_double(float(v)) for v in (start_year, perigee_km, apogee_km, inc_deg, raan_deg, argp_deg,
                                            area_to_mass, end_year)]
    life, dwell, last, msg = ct.c_double(), ct.c_double(), ct.c_int(), ct.c_int()
    with _in_dir(das_home()):
        lib.LIFEDWELL(*[ct.byref(a) for a in args], ct.byref(life), ct.byref(dwell), ct.byref(last), ct.byref(msg))
    return life.value, dwell.value, msg.value
