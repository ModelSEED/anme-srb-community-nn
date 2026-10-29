#!/usr/bin/env python3
"""
nn_sample_fba.py — Song et al. 2025 (doi:10.1038/s41598-025-89997-9) training-point
generator for the ANME-SRB community surrogate NN.

Inputs sampled (uptake-rate CAPS, mmol/gDW/hr), random-uniform over [0, max]:
    CH4 (cpd01024), SO4 (cpd00048), NH3 (cpd00013)
Sampling maxima span the model's DYNAMIC (limiting->saturating) range (from nn_sat_sweep);
the environment can supply at least this much (the diffusive ceiling is far higher, i.e.
diffusion is non-limiting — see docs/NN_surrogate_v1.md).

Per sample: MULTI-STEP (lexicographic) FBA:
    (1) maximize community biomass bio1            -> growth mu
    (2) fix bio1 >= (1-eps)*mu, maximize sulfide   -> realistic max sulfide secretion
    (3) fix sulfide, maximize CO2                  -> realistic max CO2 secretion
Outputs: growth, sulfide, CO2, and REALIZED uptakes of CH4/SO4/NH3/N2 (realized uptake
sits below its cap once another constraint binds; realized N2 = the ammonia->N2-fixation
switch). Regenerates data/training_data.csv.
"""
import os, json, csv, numpy as np, cobra
from cobra.io import load_json_model
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import MODEL, DATADIR
cobra.Configuration().solver = "glpk"

EX = {"CH4": "EX_cpd01024_e0", "SO4": "EX_cpd00048_e0", "NH3": "EX_cpd00013_e0"}
SULFIDE, CO2, N2, BIO = "EX_cpd00239_e0", "EX_cpd00011_e0", "EX_cpd00528_e0", "bio1"
RANGES = {"CH4": (0.0, 300.0), "SO4": (0.0, 200.0), "NH3": (0.0, 70.0)}
N = int(os.environ.get("NN_SAMPLES", "3000"))
SEED = int(os.environ.get("NN_SEED", "20261028"))
EPS = 1e-6

def fresh():
    m = load_json_model(MODEL); m.solver = "glpk"
    # lift artificial 100-unit secretion caps so lexicographic max reflects mass balance
    for ex in (SULFIDE, CO2):
        m.reactions.get_by_id(ex).upper_bound = 1000.0
    # open N2 uptake (the nitrogen fallback) in case a base model ships it closed; the
    # committed model already has it open.
    m.reactions.get_by_id(N2).lower_bound = -1000.0
    return m

def lexicographic(m, caps):
    with m:                       # cobra snapshot+restore of bounds/objective per sample
        for k, ex in EX.items():
            m.reactions.get_by_id(ex).lower_bound = -float(caps[k])
        m.objective = BIO; m.objective.direction = "max"
        s1 = m.optimize()
        if s1.status != "optimal" or s1.objective_value is None or s1.objective_value < 1e-7:
            return dict(growth=0.0, sulfide=0.0, co2=0.0, CH4_real=0.0, SO4_real=0.0,
                        NH3_real=0.0, N2_real=0.0, feasible=int(s1.status == "optimal"))
        mu = s1.objective_value
        m.reactions.get_by_id(BIO).lower_bound = mu * (1 - EPS)
        m.objective = SULFIDE; m.objective.direction = "max"
        s2 = m.optimize()
        sulf = s2.fluxes[SULFIDE] if s2.status == "optimal" else s1.fluxes[SULFIDE]
        if s2.status == "optimal":
            m.reactions.get_by_id(SULFIDE).lower_bound = sulf * (1 - EPS) if sulf > 0 else sulf
        m.objective = CO2; m.objective.direction = "max"
        s3 = m.optimize()
        final = s3 if s3.status == "optimal" else (s2 if s2.status == "optimal" else s1)
        return dict(growth=float(mu), sulfide=float(final.fluxes[SULFIDE]),
                    co2=float(final.fluxes[CO2]), CH4_real=float(-final.fluxes[EX["CH4"]]),
                    SO4_real=float(-final.fluxes[EX["SO4"]]), NH3_real=float(-final.fluxes[EX["NH3"]]),
                    N2_real=float(-final.fluxes[N2]), feasible=1)

def main():
    rng = np.random.default_rng(SEED)
    caps_all = {k: rng.uniform(lo, hi, N) for k, (lo, hi) in RANGES.items()}
    m = fresh(); rows = []
    for i in range(N):
        caps = {k: caps_all[k][i] for k in EX}
        r = lexicographic(m, caps); r.update({f"{k}_cap": caps[k] for k in EX})
        rows.append(r)
        if (i + 1) % 250 == 0: print(f"  ... {i+1}/{N}", flush=True)
    cols = ["CH4_cap", "SO4_cap", "NH3_cap", "growth", "sulfide", "co2",
            "CH4_real", "SO4_real", "NH3_real", "N2_real", "feasible"]
    path = os.path.join(DATADIR, "training_data.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r[c] for c in cols})
    g = np.array([r["growth"] for r in rows])
    meta = dict(n=N, seed=SEED, ranges=RANGES,
                method="lexicographic FBA (max bio1 -> max sulfide -> max CO2)",
                model=os.path.basename(MODEL), n_growing=int((g > 1e-7).sum()),
                growth_min=float(g.min()), growth_max=float(g.max()))
    with open(os.path.join(DATADIR, "training_data_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("WROTE", path); print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
