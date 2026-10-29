# v1 — ANME-2a × SEEP-SRB2 AOM surrogate (2026-10-28)

First proxy-NN surrogate of the functioning ANME–SRB community model. Full write-up (unit
translation, assumptions, sampling rationale, the N₂-fixation fix, validation):
[`docs/NN_surrogate_v1.md`](docs/NN_surrogate_v1.md).

## What this version is

- **Underlying model:** `model/community_ANME2a_SRB2_AOM_final_N2open.json` — ANME-2a ×
  SEEP-SRB2 merged community (MSCommunity), AOM-coupled (methane oxidation → shared electron
  carrier → sulfate reduction), with **N₂ uptake opened so nitrogenase can fix N₂** when
  ammonia is scarce. `model/community_ANME2a_SRB2_AOM_final.json` is the same model with N₂
  uptake closed (the pre-fix original), kept for reference.
- **Surrogate:** `surrogate/nn_surrogate.joblib` — sklearn MIMO MLP (tanh, 10×5), trained on
  3000 lexicographic-FBA samples.
  - **Inputs (3):** `CH4_cap`, `SO4_cap`, `NH3_cap` — uptake-rate caps (mmol·gDW⁻¹·hr⁻¹).
  - **Outputs (6):** `growth`, `sulfide`, `co2`, `SO4_real`, `NH3_real`, `N2_real`.

## Fit (held-out test set, `surrogate/nn_metrics.json`) — overall R² = 0.9994

| output | R² | RMSE |
|---|---|---|
| growth | 0.99983 | 0.020 |
| sulfide | 0.99955 | 0.84 |
| CO₂ | 0.99822 | 0.74 |
| realized SO₄ | 0.99956 | 0.84 |
| realized NH₃ | 0.99981 | 0.18 |
| realized N₂ (fixation) | 0.99960 | 0.059 |

Differential-validated against direct FBA on the ammonia / N₂-fixation switch (see docs).

## Key facts to know when using it

- **Diffusion is non-limiting** at the source flux bounds, so sampling is over the model's
  *dynamic* response range (CH₄ 0–300, SO₄ 0–200, NH₃ 0–70 mmol·gDW⁻¹·hr⁻¹), not the raw
  diffusive caps. Predictions outside these ranges are extrapolation.
- **One load-bearing assumption:** gDW/cell = 4.37×10⁻¹¹ g (from the supplied cell
  geometry). The flux scale rescales with it; the sampling ranges do not.
- Energetics are mass-balance level (EGC-clean, **no thermodynamic/TMFA constraints**);
  GAM at the ModelSEED default of 40. Single growth option (methane-driven AOM).

## Files

| path | what |
|---|---|
| `model/community_ANME2a_SRB2_AOM_final_N2open.json` | genome-scale community model (N₂ fixation enabled) |
| `model/community_ANME2a_SRB2_AOM_final.json` | same, N₂ uptake closed (reference) |
| `data/training_data.csv` (+ `_meta.json`) | 3000-point training set |
| `data/bounds.xlsx` | source environmental flux-bounds (provenance input) |
| `surrogate/nn_surrogate.joblib` (+ `nn_metrics.json`) | trained surrogate + fit metrics |
| `code/*.py` | the full reproduce pipeline (see repo README) |
| `docs/NN_surrogate_v1.md` | the write-up |
