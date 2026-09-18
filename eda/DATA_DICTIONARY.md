# Data dictionary

Every column in the MERFISH dataset, what it means, how complete it is, and — crucially — **how
we used it**. Column meanings follow the organizer's `Data.Description.md`; the missingness and the
"role in our model" come from our own EDA and modeling decisions.

A design principle runs through the last column: **experiment-specific identities are used only for
validation, never as predictive features** — so the model can't lean on "which mouse / section /
batch a cell came from" and fail on a new experiment.

## Gene expression (the primary signal)

| Columns | Description | Type | Missing | Role in our model |
|---|---|---|---|---|
| 200 gene columns | Per-cell transcript counts for each measured gene | Integer, **very sparse** (median 12/200 genes non-zero, ~21 counts/cell) | 0% | **Primary features**, log-normalized. Sparse and noisy, so a strong prior on top helps. |

## Cell type (the target)

| Column | Description | Type | Missing | Role in our model |
|---|---|---|---|---|
| `MERFISH_cell_type_annotation` | Cell type — one of **60** | Categorical | 0% train / **100% test** (hidden) | **Prediction target.** 60 classes, highly imbalanced (703 → 3 examples). |

## Anatomical / biological features (kept)

| Column | Description | Type | Missing (train) | Role in our model |
|---|---|---|---|---|
| `Excitatory_vs_Inhibitory` | Excitatory / inhibitory (blank = non-neuronal) | Categorical | 62.8% | **Feature + hard-constraint decoder.** The 60 types split cleanly into **24 excitatory / 15 inhibitory / 21 non-neuronal with zero overlap**, so we forbid any prediction from crossing this line. |
| `Region` | Tissue-section region | Categorical | 62.8% | Feature (one-hot) + **target encoding** + input to the anatomical bucket. |
| `Segment` | Spinal segment | Categorical | 59.2% | Feature + **target encoding** + input to the anatomical bucket. |
| `AP_position` | Anterior–posterior position of the section | Categorical | 0% | Anatomical feature (rostro-caudal axis). |
| `volume` | Cell volume | Numeric | 0% | QC signal; used as within-section percentile ranks rather than raw (raw skews). |
| `Gender` | Mouse sex | Categorical | 0% | Used cautiously — a biological variable, but with 5F/5M it can act as a weak animal-identity proxy. |

*Missing E/I / Region / Segment is **informative**, not an error: it marks non-neuronal cells. We
represent missingness explicitly instead of dropping those cells.*

## Spatial coordinates

| Column | Description | Type | Missing | Role in our model |
|---|---|---|---|---|
| `center_x`, `center_y` | Cell position in the tissue | Numeric | 0% | Raw coordinates **excluded** (they skew the model — cells aren't uniformly dispersed). Neighborhood features were explored but left out of the submission (see lessons learned). |

## Experiment identifiers (used for validation, **excluded from features**)

| Column | Description | Type | Missing | Role in our model |
|---|---|---|---|---|
| `Mouse_ID` | Source mouse (10 total) | Categorical | 0% | **Not a feature.** Grouping key for **leave-one-mouse-out** validation. |
| `Datasets` | Batch / dataset ID | Categorical | 0% | **Not a feature.** Grouping key for **leave-one-batch-out** validation. |
| `Section_ID` | Tissue section (108 total) | Categorical | 0% | **Not a feature.** Used for within-section QC percentile ranks and as a grouping unit. |
| Cell ID (row label) | Unique cell identifier | String | 0% | Key only; joins predictions to `meta_test.csv` order. |

## Dataset at a glance

- **10,000 cells** (≈5,000 train / 5,000 test), **200 genes**, **60 cell types**.
- **10 mice**, **108 tissue sections**; train and test are drawn from the *same* mice/sections and
  are spatially interleaved → random CV is optimistic (hence group validation).
- Expression is **~93% zeros** (median 12 non-zero genes, ~21 counts per cell).
- Class imbalance ≈ **234×** (largest class 703 vs. smallest 3).
