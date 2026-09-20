# Presentation QA report

## Delivered artifact

- `green_roof_pitch.pptx`
- 19 slides: 12-slide main pitch, one appendix divider, and six Q&A slides
- 19 speaker-note pages with `[Sources]` blocks
- 9 embedded media assets

## Checks performed

- Rendered and visually inspected all 19 authoring previews slide by slide.
- Re-rendered the exported PPTX with the presentation renderer.
- Rendered the final deck with Microsoft PowerPoint and inspected the slides that
  previously showed title-reflow risk.
- Ran `slides_test.py`: **passed; no overflow detected**.
- Ran `audit_pptx_quality.py --fail-on high`: **high=0, medium=1**.
- Verified that the deck reports validation findings only and keeps test results
  explicitly unclaimed.

## Corrective revisions

- Normalized 0/1 binary masks to 0/255 for correct visual display.
- Increased the preprocessing evidence images and reorganized the five GIS
  stages into a vertical sequence.
- Added title safe margins and shortened the longest appendix title to avoid
  PowerPoint text reflow.
- Restored section labels and title rules on slides affected by PowerPoint's
  layout import.

## Remaining audit item

- The one medium finding is an intentional cover-image crop on slide 1. The
  image is a contextual aerial tile without axes, legend, panel labels, or scale
  bars; the crop does not remove evidence required for any quantitative claim.

## Evidence boundary

- Validation metrics and thresholds are sourced from the two validation-analysis
  summaries.
- The test set remains untouched.
- The postcode boundary is current official geometry and is not verified as the
  historical 2016 boundary.

