# Frozen public legacy baseline

The immutable baseline is upstream commit
`659ce323acf2c4cc62e697a03c5d90fc809aa9b7`, retained on GitHub as branch
`legacy/upstream-659ce323`. The branch must continue to point to that exact
commit; maintained changes live on other branches.

`manifest.json` records the repository, commit, legacy branch, and hashes of the
two reference model sources. The reference files are byte-identical to the
corresponding files in the frozen commit and exist only so numerical-equivalence
tests can run offline. Do not import them from production code.

The full source snapshot remains reproducible from Git itself:

```bash
git archive --format=zip --output=SegNeuron-659ce323.zip 659ce323acf2c4cc62e697a03c5d90fc809aa9b7
```

`tests/test_legacy_contract.py` verifies the manifest and reference hashes;
`tests/test_model_equivalence.py` verifies model state keys and exact forward
outputs against those frozen sources.

## Intentionally preserved behavior

- MNet architecture, parameter names, and checkpoint key `model_weights`.
- Input and output tensor shapes and `(z, y, x)` axis convention.
- Three affinity channels and one boundary channel.
- Legacy `/ 255.0` normalization.
- Sliding-window crop, stride, padding, and blending behavior.
- Legacy loss-name mapping (`MSELoss` selects `WeightedBCE`; `BCELoss` selects
  `WeightedMSE`).
- Existing FRMC watershed and multicut behavior.

These behaviors may be surprising, but changing them requires a separate,
explicitly benchmarked `corrected` mode and golden data.

## Stabilization fixes outside the legacy branch

- Removed references to the undefined `valid_provider` in both training entry
  points.
- Corrected the supervised `loop` argument count.
- Prevented intermittent pretraining dataset-index overflow.
- Added the missing `MODEL.model_type` field with the existing code's
  `superhuman` default.
- Added fail-fast configuration validation.
- Added a reproducible environment description and compatibility tests.
