#!/usr/bin/env python3
"""
nn_bounds_convert.py — translate the per-cell diffusive flux caps in data/bounds.xlsx
into FBA uptake caps (mmol/gDW/hr).

The bounds file computes a MAX diffusive supply flux per cell via Fick's law across a
boundary layer:  F = D * (Cenv - 0) / Ldbl  [mol/m2/s];  per-cell = F * A_cell.
We (a) recompute that per-cell flux from the physical inputs (so we do not depend on the
spreadsheet's own derived column, which carries a 1e6 unit slip), (b) derive gDW/cell from
the supplied cell geometry, (c) convert to mmol/gDW/hr. Everything load-bearing is a NAMED
CONSTANT so it is trivial to rescale.
"""
import math, json, os
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATADIR

LDBL_M    = 1e-6                     # boundary layer thickness (m)
A_CELL_M2 = 1.2566370614359175e-10   # cell surface area (m2)

INPUTS = {
    "CH4": dict(cpd="cpd01024", D=8.97e-10, Cmax_native=100000.0, conv=0.001),  # uM -> mol/m3
    "SO4": dict(cpd="cpd00048", D=5.72e-10, Cmax_native=30.0,     conv=1.0),    # mM -> mol/m3
    "NH3": dict(cpd="cpd00013", D=1.10e-09, Cmax_native=5000.0,   conv=0.001),  # uM -> mol/m3
}

# gDW/cell (the one load-bearing biological assumption): sphere from the surface area,
# wet volume * wet density * dry fraction.
WET_DENSITY_G_PER_L = 1100.0   # ~1.1 g/mL
DRY_FRACTION        = 0.30     # dry weight ~30% of wet mass

def gdw_per_cell():
    r_m = math.sqrt(A_CELL_M2 / (4.0 * math.pi))
    V_L = (4.0/3.0) * math.pi * r_m**3 * 1000.0
    return r_m, V_L, V_L * WET_DENSITY_G_PER_L * DRY_FRACTION

def percell_flux_mmol_per_hr(D, Cmax_native, conv):
    F = D * (Cmax_native * conv) / LDBL_M            # mol/m2/s
    return F, F * A_CELL_M2 * 3600.0 * 1000.0        # -> mmol/cell/hr

if __name__ == "__main__":
    # (bounds.xlsx is kept in data/ as the provenance input; the physical constants above
    #  are transcribed from it. Reading the workbook is optional — openpyxl if present.)
    r_m, V_L, gdw = gdw_per_cell()
    print(f"# bounds.xlsx present: {os.path.exists(os.path.join(DATADIR,'bounds.xlsx'))}")
    print(f"# cell: r={r_m*1e6:.3f} um, V={V_L*1e15:.3f} fL, gDW/cell={gdw:.3e} g "
          f"(wet_density={WET_DENSITY_G_PER_L} g/L, dry_frac={DRY_FRACTION})\n")
    print(f"{'input':6}{'cpd':11}{'F[mol/m2/s]':>14}{'ub[mmol/cell/hr]':>18}{'ub[mmol/gDW/hr]':>18}")
    caps = {}
    for name, p in INPUTS.items():
        F, f_cell = percell_flux_mmol_per_hr(p["D"], p["Cmax_native"], p["conv"])
        ub = f_cell / gdw
        caps[name] = dict(cpd=p["cpd"], ub_mmol_gDW_hr=ub)
        print(f"{name:6}{p['cpd']:11}{F:14.4e}{f_cell:18.4e}{ub:18.1f}")
    print("\n# NOTE: diffusive caps >> model demand -> diffusion non-limiting; sampling is")
    print("#       over the model's dynamic range (see nn_sample_fba.py RANGES).")
