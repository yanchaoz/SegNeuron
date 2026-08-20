# Contributing to the legacy-stabilized codebase

Treat the pinned upstream commit and reference sources documented in `legacy/` as the behavioral baseline. Keep formatting and deterministic defect fixes separate from changes to models, losses, augmentation, normalization, inference, or FRMC postprocessing.

Before submitting a change, run:

```bash
python -m ruff check .
python -B -m unittest discover -s tests -v
```

If a change can affect numerical behavior, add a focused golden test and document the expected difference. Do not update a golden result merely to make a failing test pass; first explain and review why the output should change.

The utility modules mirrored under `Pretrain/utils` and `Train_and_Inference/utils` must remain identical except for the intentionally different `show.py`. The contract test enforces this until the duplicated code can be migrated safely to one shared package.
