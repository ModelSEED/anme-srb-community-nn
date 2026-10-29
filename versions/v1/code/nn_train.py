#!/usr/bin/env python3
"""
nn_train.py — train the proxy MIMO neural network on the lexicographic-FBA dataset
(Song et al. 2025).

INPUTS  (3): CH4_cap, SO4_cap, NH3_cap          (imposed uptake upper limits)
OUTPUTS (6): growth, sulfide, co2, SO4_real, NH3_real, N2_real
             (biomass + byproduct secretions + realized consumptions; N2_real is the
             ammonia->N2-fixation switch). Writes surrogate/nn_surrogate.joblib + metrics.
"""
import os, json, numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATADIR, SURRDIR

INPUTS = ["CH4_cap", "SO4_cap", "NH3_cap"]
OUTPUTS = ["growth", "sulfide", "co2", "SO4_real", "NH3_real", "N2_real"]
SEED = 20261028

def main():
    df = pd.read_csv(os.path.join(DATADIR, "training_data.csv"))
    X = df[INPUTS].to_numpy(float); Y = df[OUTPUTS].to_numpy(float)
    n = len(df); rng = np.random.default_rng(SEED); idx = rng.permutation(n)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    ys = StandardScaler().fit(Y[tr])
    pipe = Pipeline([
        ("xs", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(10, 10, 10, 10, 10), activation="tanh",
                             solver="lbfgs", alpha=1e-2, max_iter=5000, random_state=SEED)),
    ])
    pipe.fit(X[tr], ys.transform(Y[tr]))

    def evalset(ix):
        pred = ys.inverse_transform(pipe.predict(X[ix])); per = {}
        for j, name in enumerate(OUTPUTS):
            per[name] = dict(r2=float(r2_score(Y[ix, j], pred[:, j])),
                             rmse=float(np.sqrt(mean_squared_error(Y[ix, j], pred[:, j]))))
        per["_overall_r2"] = float(r2_score(Y[ix], pred)); return per

    metrics = dict(n=n, split=dict(train=len(tr), val=len(va), test=len(te)),
                   inputs=INPUTS, outputs=OUTPUTS,
                   topology="MLP tanh (10,10,10,10,10) lbfgs alpha=1e-2",
                   train=evalset(tr), val=evalset(va), test=evalset(te))
    os.makedirs(SURRDIR, exist_ok=True)
    joblib.dump(dict(pipe=pipe, yscaler=ys, inputs=INPUTS, outputs=OUTPUTS),
                os.path.join(SURRDIR, "nn_surrogate.joblib"))
    with open(os.path.join(SURRDIR, "nn_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
