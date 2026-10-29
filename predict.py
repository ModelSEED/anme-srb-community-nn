#!/usr/bin/env python3
"""
predict.py — load a version's trained surrogate and predict community behavior from the
three uptake caps. No FBA solver needed (pure sklearn inference).

Usage:
    python predict.py --ch4 150 --so4 120 --nh3 5
    python predict.py --version v1 --ch4 100 --so4 60 --nh3 0     # 0 ammonia -> N2 fixation

Inputs are uptake-rate caps in mmol/gDW/hr. Prints growth, sulfide/CO2 secretion, and the
realized sulfate/ammonia/N2 uptakes (realized N2 > 0 means the community is fixing N2).
"""
import os, glob, argparse, numpy as np, joblib

HERE = os.path.dirname(os.path.abspath(__file__))

def latest_version():
    vs = sorted(glob.glob(os.path.join(HERE, "versions", "v*")),
                key=lambda p: int(os.path.basename(p)[1:]) if os.path.basename(p)[1:].isdigit() else -1)
    if not vs:
        raise SystemExit("no versions/ found")
    return os.path.basename(vs[-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None, help="version dir (default: latest)")
    ap.add_argument("--ch4", type=float, required=True)
    ap.add_argument("--so4", type=float, required=True)
    ap.add_argument("--nh3", type=float, required=True)
    a = ap.parse_args()
    ver = a.version or latest_version()
    bundle = joblib.load(os.path.join(HERE, "versions", ver, "surrogate", "nn_surrogate.joblib"))
    pipe, ys, OUT = bundle["pipe"], bundle["yscaler"], bundle["outputs"]
    y = ys.inverse_transform(pipe.predict(np.array([[a.ch4, a.so4, a.nh3]], float)))[0]
    pred = {k: round(float(v), 4) for k, v in zip(OUT, y)}
    # clip tiny negative/near-zero regression artifacts in the realized-uptake outputs
    for k in ("SO4_real", "NH3_real", "N2_real"):
        if k in pred and pred[k] < 0: pred[k] = 0.0
    print(f"version={ver}  inputs: CH4={a.ch4} SO4={a.so4} NH3={a.nh3} (mmol/gDW/hr)")
    for k in OUT:
        print(f"  {k:10} = {pred[k]}")
    if pred.get("N2_real", 0) > 0.1:
        print("  -> N2 fixation active (ammonia insufficient; nitrogenase supplying N)")

if __name__ == "__main__":
    main()
