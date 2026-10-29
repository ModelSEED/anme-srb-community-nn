#!/usr/bin/env python3
"""
nn_validate.py — differential test of the surrogate against direct FBA along the ammonia
axis (the N2-fixation switch), at fixed CH4/SO4.
"""
import numpy as np, joblib, os, cobra
from cobra.io import load_json_model
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import MODEL, SURRDIR
cobra.Configuration().solver = "glpk"

EX = {"CH4": "EX_cpd01024_e0", "SO4": "EX_cpd00048_e0", "NH3": "EX_cpd00013_e0"}
N2, SULFIDE, CO2, BIO = "EX_cpd00528_e0", "EX_cpd00239_e0", "EX_cpd00011_e0", "bio1"

bundle = joblib.load(os.path.join(SURRDIR, "nn_surrogate.joblib"))
pipe, ys, INP, OUT = bundle["pipe"], bundle["yscaler"], bundle["inputs"], bundle["outputs"]

def nn_predict(ch4, so4, nh3):
    y = ys.inverse_transform(pipe.predict(np.array([[ch4, so4, nh3]], float)))[0]
    return dict(zip(OUT, y))

def fba(ch4, so4, nh3):
    m = load_json_model(MODEL); m.solver = "glpk"
    for ex in (SULFIDE, CO2): m.reactions.get_by_id(ex).upper_bound = 1000.0
    m.reactions.get_by_id(N2).lower_bound = -1000.0
    m.reactions.get_by_id(EX["CH4"]).lower_bound = -ch4
    m.reactions.get_by_id(EX["SO4"]).lower_bound = -so4
    m.reactions.get_by_id(EX["NH3"]).lower_bound = -nh3
    m.objective = BIO; s = m.optimize()
    return dict(growth=s.objective_value, N2_real=-s.fluxes[N2], NH3_real=-s.fluxes[EX["NH3"]])

if __name__ == "__main__":
    CH4, SO4 = 150.0, 120.0
    print(f"Ammonia sweep at CH4={CH4}, SO4={SO4}  (NN vs FBA)")
    print(f"{'NH3cap':>7} | {'growth_NN':>9} {'growth_FBA':>10} | {'N2_NN':>7} {'N2_FBA':>7} | {'NH3r_NN':>8} {'NH3r_FBA':>8}")
    for nh3 in [0, 1, 3, 5, 10, 20, 30, 50, 70]:
        p = nn_predict(CH4, SO4, nh3); f = fba(CH4, SO4, nh3)
        print(f"{nh3:7.1f} | {p['growth']:9.3f} {f['growth']:10.3f} | "
              f"{p['N2_real']:7.3f} {f['N2_real']:7.3f} | {p['NH3_real']:8.3f} {f['NH3_real']:8.3f}")
