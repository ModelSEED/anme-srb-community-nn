# First proxy-NN surrogate of the ANME-SRB community model — v1

**Date:** 2026-10-28 · **Arc:** anme-srb-community-modeling · **Model:**
`build/models/community_ANME2a_SRB2_AOM_final.json` (ANME-2a × SEEP-SRB2, AOM-coupled) ·
**Method:** Song, Ahamed, Lee, **Henry** et al. 2025, *Sci Rep* (doi:10.1038/s41598-025-89997-9)

This is the first surrogate neural network trained on the functioning community model. It maps imposed
**uptake-rate caps** of methane, sulfate, and ammonia to community growth and the key secretion fluxes, using the
multi-step-FBA → ANN recipe developed with Hyun-Seob Song.

## 1. Flux-range translation from Christof's bounds file (per-cell → per-gram-biomass)

Christof's `external-context/bounds.xlsx` gives, per compound, an environmental concentration range and a maximum
**diffusive supply flux per cell** computed by Fick's law across a boundary layer:

```
F = D · (C_env − 0) / L_dbl        [mol·m⁻²·s⁻¹]      (L_dbl = 1 µm boundary layer)
per-cell flux = F · A_cell         [mol·cell⁻¹·s⁻¹]   (A_cell = 1.2566×10⁻¹⁰ m²)
```

FBA needs **mmol·gDW⁻¹·hr⁻¹**, so we recompute the per-cell flux from the physical inputs (diffusion coefficient,
max concentration, boundary-layer thickness, cell area — script `nn_bounds_convert.py`) and divide by a dry weight
per cell derived from the cell geometry Christof supplied:

- **gDW/cell = 4.37×10⁻¹¹ g.** From the given cell surface area, treating the cell as a sphere (radius 3.16 µm,
  volume 132 fL), with wet density 1100 g/L and dry fraction 0.30. *These last two are standard microbiology
  constants and are the single load-bearing assumption in the translation — the caps scale inversely with gDW/cell,
  so correct this one number and everything rescales.*

| input | cpd | max conc (env) | diffusive cap **mmol·gDW⁻¹·hr⁻¹** |
|---|---|---|---|
| CH₄ | cpd01024 | 100 mM | **≈ 9.3×10⁵** |
| SO₄ | cpd00048 | 30 mM | **≈ 1.8×10⁵** |
| NH₃ | cpd00013 | 5 mM (as NH₄⁺) | **≈ 5.7×10⁴** |

**Two things found during the translation, both reported honestly:**

1. **The spreadsheet's own "mmol/cell/hr" column is 10⁶× too small** (a unit slip). Our recomputation reproduces
   Christof's mol·m⁻²·s⁻¹ column *exactly*, so the physics is right; the physically-correct per-cell methane cap is
   4.06×10⁻⁵ mmol·cell⁻¹·hr⁻¹, not the 4.06×10⁻¹¹ in the sheet. We used the recomputed value.
2. **Diffusion is not the limiting process at this geometry.** The diffusive caps above are thousands of times
   larger than anything the model can metabolize (see §2). So the environment can supply methane/sulfate/ammonia
   far faster than the community uses them; the realistic operating point is set by the organisms' stoichiometry,
   not by diffusive supply.

## 2. What the model actually responds to (sets the sampling ranges)

Single-variable sweeps (`nn_sat_sweep.py`) on the final model, each input varied with the others replete:

- **Methane** — strictly linear growth driver: bio1 ≈ 0.0227 × (methane uptake), limiting from 0 up to a realized
  uptake of **≈ 263**, then the curve plateaus (an internal limit binds at high methane).
- **Sulfate** — co-limits only in the sub-stoichiometric band: realized sulfate = **0.586 × realized methane**;
  above that, growth saturates (methane-limited).
- **Ammonia** — with N₂ fixation enabled (below), ammonia is **not hard-limiting**: as the ammonia cap falls, the
  community progressively switches from ammonia assimilation to **N₂ fixation** (nitrogenase `rxn06874`), at a
  growth penalty. At fixed CH₄=150/SO₄=120, growth falls gently from 3.40 (ammonia replete) to 2.11 (ammonia zero,
  all N from fixation ≈ 62% of the replete rate — nitrogenase's 16-ATP/N₂ cost), rather than collapsing to zero.

### Nitrogen fixation must be switched on (model fix)

The final model carries functional nitrogenase in **both** members, but ships with the **N₂ exchange closed for
uptake** (`EX_cpd00528_e0` lower bound 0 — N₂ could only be secreted). Left as-is, zero ammonia gives zero growth
instead of inducing fixation — defeating the purpose of the ammonia axis. The fix is to **open N₂ uptake**
(N₂ is environmentally always available). We persisted this as a durable variant
`build/models/community_ANME2a_SRB2_AOM_final_N2open.json` (original preserved) and the sampler opens N₂ at load.
Opening N₂ does **not** perturb the ammonia-replete optimum: with ammonia available the community still prefers
assimilation and fixes no N₂ (assimilation is cheaper), so the switch is induced only by ammonia scarcity.

Because the diffusive caps sit far above these saturation points, sampling uniformly over the raw diffusive range
would place essentially every point in the flat, saturated regime and teach the surrogate nothing. We therefore
sample each input over its **dynamic (limiting→saturating) range**, which the environment can physically supply
(the diffusive ceiling is far higher):

| input | sampling range (mmol·gDW⁻¹·hr⁻¹) | why |
|---|---|---|
| CH₄ | 0 – 300 | spans the full linear-rise (0–263) + plateau |
| SO₄ | 0 – 200 | covers co-limitation up to 0.586 × 300 ≈ 176 |
| NH₃ | 0 – 70 | covers co-limitation up to 0.197 × 263 ≈ 52 |

## 3. Training-data generation (Song method)

`nn_sample_fba.py`: N = 3000 points, each input drawn independently **random-uniform** over its range (seed
20261028). Per point, a **multi-step lexicographic FBA** (not plain pFBA):

1. maximize community biomass `bio1` → growth μ;
2. fix `bio1 ≥ (1−ε)·μ`, **maximize sulfide secretion** (the primary AOM product);
3. fix sulfide, **maximize CO₂ secretion**.

This reproduces realistic byproduct secretion instead of the flux-minimized pFBA solution that suppresses overflow.
(The artificial 100-unit secretion caps on the H₂S and CO₂ exchanges were lifted to 1000 so the maximization is
bounded by mass balance — sulfide tracks ~1:1 with the methane oxidized — not by an arbitrary clip.)

- **NN inputs (3):** CH₄ cap, SO₄ cap, NH₃ cap (the imposed uptake upper limits).
- **NN outputs (6):** growth, sulfide secretion, CO₂ secretion (products for RTM source terms), and the **realized
  uptakes** of sulfate, ammonia, and N₂ — kept as outputs because a realized uptake drops below its cap once
  another constraint binds (Song's limit/role signal). **Realized N₂ uptake is the ammonia→N₂-fixation switch**:
  ≈ 0 when ammonia is replete, rising as ammonia falls.

## 4. The surrogate NN

`nn_train.py`: sklearn `MLPRegressor`, tanh, hidden layers (10,10,10,10,10) ≈ Song's ~10-node × 5-layer topology,
inputs and outputs standardized, 70/15/15 train/val/test split (3000 points, all feasible, growth 0.003–5.77).
(sklearn has no Bayesian-regularization backprop; we approximate Song's regularization with L2 + the lbfgs solver.)

**Held-out test-set fit (`build/nn/nn_metrics.json`), overall R² = 0.9994:**

| output | R² (test) | RMSE (test) |
|---|---|---|
| growth | 0.99983 | 0.020 |
| sulfide | 0.99955 | 0.84 |
| CO₂ | 0.99822 | 0.74 |
| realized SO₄ | 0.99956 | 0.84 |
| realized NH₃ | 0.99981 | 0.18 |
| realized N₂ (fixation) | 0.99960 | 0.059 |

The response surface is piecewise-linear (an LP), so the small net fits it tightly — as expected; the surrogate's
value is speed inside an RTM loop, not capturing nonlinearity that isn't there (see caveats).

### Differential validation against FBA — the ammonia / N₂-fixation switch (`nn_validate.py`)

At fixed CH₄=150, SO₄=120, sweeping the ammonia cap, surrogate vs direct FBA:

| NH₃ cap | growth NN | growth FBA | N₂-fix NN | N₂-fix FBA |
|---|---|---|---|---|
| 0  | 2.113 | 2.110 | 9.25 | 9.16 |
| 5  | 2.354 | 2.344 | 7.69 | 7.68 |
| 10 | 2.597 | 2.578 | 6.17 | 6.19 |
| 20 | 3.045 | 3.047 | 3.30 | 3.23 |
| 30 | 3.410 | 3.401 | 0.13 | 0.00 |
| 70 | 3.406 | 3.401 | 0.01 | 0.00 |

The surrogate reproduces the smooth switch-over: full N₂ fixation at zero ammonia, declining as ammonia rises, and
shutting off above ~30 where ammonia becomes replete and growth saturates. (Tiny non-zero N₂ / negative NH₃
predictions in the replete regime are sub-0.2 regression smoothing artifacts; clip at 0 for use.)

## 5. Caveats

- The surrogate inherits every caveat of the underlying model: mass-balance (EGC-clean) energetics, **no ΔG/TMFA
  constraints**, GAM at the ModelSEED default of 40. Yields are stoichiometric, not thermodynamic.
- The response surface is piecewise-linear (an LP), so a small net fits it easily; the value of the surrogate is
  speed inside an RTM loop, not capturing nonlinearity that isn't there. Nonlinearity will appear once kinetics /
  thermodynamics / multiple growth options are added (Song's cybernetic layer).
- `gDW/cell` and the derived caps are only as good as the sphere+density+dry-fraction assumption in §1; the whole
  flux scale moves with it. The sampling ranges in §2, by contrast, are set from the model's own response and are
  robust to that assumption.
- Single growth option (methane-driven AOM). If the seep has appreciable acetate/H₂, a second (acetate-fed) option
  + Song's cybernetic arbitration would be the next extension.
