# ANME–SRB community surrogate models

Versioned surrogate (proxy) **neural-network models** of a genome-scale **ANME–SRB
syntrophic community** (anaerobic oxidation of methane coupled to sulfate reduction), built
with the FBA → ANN method of **Song, Ahamed, Lee, Henry et al. 2025**
([*Sci Rep* 15:6042](https://doi.org/10.1038/s41598-025-89997-9)).

Each surrogate maps imposed substrate **uptake caps** (methane, sulfate, ammonia) to
community **growth**, **byproduct secretion** (sulfide, CO₂), and **realized consumption**
(sulfate, ammonia, and **N₂ fixation**) — fast enough to embed in a reactive-transport
model while reproducing the genome-scale model's behavior.

## Repository layout

```
predict.py              # load a version's surrogate and predict from the 3 uptake caps
requirements.txt
versions/
  v1/
    model/              # the genome-scale community model(s) the surrogate was built on
    data/               # training dataset (+ meta) and the source flux-bounds file
    surrogate/          # the trained NN (joblib) + fit metrics (json)
    code/               # the full pipeline that produced this version
    docs/               # the write-up for this version
```

## Versioning

Each surrogate is a **self-contained directory** `versions/vN/` holding its own model,
data, code, trained net, and docs — so every version is reproducible on its own and old
versions never change when a new one is added. To add a version: copy the latest `vN/`
to `v{N+1}/`, change the model/data/code, retrain, and update that version's `docs/` and
the index below. Git tags (`v1`, `v2`, …) and GitHub Releases can mark each as a release.
`predict.py` defaults to the highest-numbered version; pass `--version vN` to pin one.

| version | date | model | inputs → outputs | test R² | notes |
|---|---|---|---|---|---|
| [v1](versions/v1/) | 2026-10-28 | ANME-2a × SEEP-SRB2 (AOM, N₂-fixation enabled) | CH₄/SO₄/NH₃ caps → growth, sulfide, CO₂, realized SO₄/NH₃/N₂ | 0.9994 | first surrogate; captures the ammonia→N₂-fixation switch |

## Quickstart — predict (no solver needed)

```bash
pip install -r requirements.txt
python predict.py --ch4 150 --so4 120 --nh3 5
python predict.py --ch4 100 --so4 60 --nh3 0     # zero ammonia → induces N₂ fixation
```

Inputs are uptake-rate caps in **mmol·gDW⁻¹·hr⁻¹**. Output `N2_real > 0` means the
community is fixing N₂ because ammonia is insufficient.

## Reproduce a version from scratch

From `versions/vN/code/` (needs the modeling stack — see `requirements.txt`):

```bash
cd versions/v1/code
python nn_bounds_convert.py     # environmental flux-bounds → mmol/gDW/hr caps
python nn_sat_sweep.py          # find the model's dynamic response ranges
python nn_sample_fba.py         # generate the training set (lexicographic multi-step FBA)
python nn_train.py              # train the surrogate → ../surrogate/
python nn_validate.py           # differential test vs direct FBA (the N₂-fixation switch)
```

Each version's `docs/` explains its specifics, assumptions, and validation.

## Provenance & method

The genome-scale ANME–SRB community model and the methodology are documented per version in
`versions/vN/docs/`. The surrogate method (multi-step lexicographic FBA training points →
MIMO neural net) follows Song et al. 2025. Built as part of ongoing ANME–SRB community
modeling work on the KBase/ModelSEED platform.
