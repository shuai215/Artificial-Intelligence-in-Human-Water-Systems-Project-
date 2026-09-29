# Reference-annotation quality audit

The audit establishes the correct interpretation of the evaluation target:
the supplied polygons are reference annotations, not error-free physical
ground truth. Before formal training, 50 positive tiles were selected by
coverage-stratified random sampling with seed 42 and compared with overlays of
the rasterized labels. All selected tile IDs occur in the frozen PLZ 12357
manifests.

The manual review identified qualitative examples of:

- indistinct or ambiguous label boundaries;
- local spatial offsets between visible roof surfaces and polygons;
- visible green vegetation without a corresponding annotation; and
- ambiguity in the operational definition of green-roof vegetation.

These observations show that the label quality is not sufficient for treatment
as definitive ground truth. The supplied polygons nevertheless provide one
consistent reference target across every controlled experiment, enabling fair
comparisons among training policies, losses, and architectures. Reported
metrics therefore quantify agreement with the same reference annotations. The
coverage-stratified review characterizes recurring annotation modes; it is not
a prevalence estimate, so no unsupported defect percentage is reported.

Sources:

- `../../Pre-processing/label_quality_check/selection_summary.json`
- `../../Pre-processing/label_quality_check/review.csv`
- `../../data/processed/green_roofs_2016_plz12357/manifests/`
