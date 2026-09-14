# Data — how to obtain it

The datasets are **not committed** to this repository. The competition data belongs to the
organizers, and the atlas is a separately-licensed public dataset. Download them yourself and
place them here.

## 1. Competition data (required for `src/`)

From the organizer's repository:
**Rochester-Biomedical-DS / Hackathon-Summer-2026** (`data/` folder).
See its `Data.Description.md` for column definitions.

Expected layout:

```
data/
  counts_train.csv    # cells × 200 genes (integer transcript counts)
  counts_test.csv
  meta_train.csv      # cell metadata + MERFISH_cell_type_annotation (labels)
  meta_test.csv
```

Facts: ~10,000 cells total (≈5K train / 5K test), 200 genes, ~93% zeros, 60 cell types.
Metric: overall accuracy.

## 2. Public spinal-cord atlas (only for `experiments/atlas_loso_testbed.py`)

A larger public MERFISH mouse spinal-cord atlas, used **post-competition** as a data-scaling
testbed (whole-mouse holdout).

- Zenodo record **18039571** — file `MERFISH_spinal_cord_resolved_0718.h5ad`
  (~147K cells, 500 genes, 10 mice). Cite the dataset's own DOI/license.

Place it at `reference/MERFISH_spinal_cord_resolved_0718.h5ad` (or edit `ATLAS_PATH` in the
script) and confirm the exact DOI/citation on the Zenodo record before use.

> ⚠️ This atlas overlaps the competition cells — it is used here **only** for a post-competition
> scaling analysis with strict whole-mouse holdout, never for the competition submission.
