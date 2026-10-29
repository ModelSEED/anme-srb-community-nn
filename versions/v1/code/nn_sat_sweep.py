#!/usr/bin/env python3
"""
nn_sat_sweep.py — find the model's RESPONSE range for CH4, SO4, NH3 by 1-D sweeps, to set
the sampling ranges in nn_sample_fba.py. Each input is varied with the others at default.
"""
import cobra
from cobra.io import load_json_model
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import MODEL
cobra.Configuration().solver = "glpk"

EX = {"CH4": "EX_cpd01024_e0", "SO4": "EX_cpd00048_e0", "NH3": "EX_cpd00013_e0"}

def load():
    m = load_json_model(MODEL); m.solver = "glpk"; m.objective = "bio1"
    return m

m = load(); base = m.optimize()
print("baseline bio1=%.4f" % base.objective_value)
for k, ex in EX.items():
    r = m.reactions.get_by_id(ex)
    print(f"  {k:4} {ex}: default lb={r.lower_bound}  realized={base.fluxes[ex]:.3f}")
print()
grid = [0, 1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 500]
for k, ex in EX.items():
    print(f"=== sweep {k} ({ex}) ===")
    print(f"{'cap':>6} {'bio1':>10} {'realized':>12}")
    for cap in grid:
        m2 = load(); m2.reactions.get_by_id(ex).lower_bound = -float(cap)
        s = m2.optimize()
        gv = s.objective_value if s.status == "optimal" else float("nan")
        rz = s.fluxes[ex] if s.status == "optimal" else float("nan")
        print(f"{cap:6d} {gv:10.4f} {rz:12.3f}")
    print()
