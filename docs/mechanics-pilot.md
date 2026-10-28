# Event-aligned lead-knee image-pattern pilot

This is an **exploratory 2D projection report**, not a validated anatomical 3D
mechanics report. The main pitch-record statistics continue to use all acquired
records, independently of this small motion sample.

## Scope and events

The pilot contains six automatically linked, distinct pitches by Choi Won-tae
(`65320`): three fastballs and three sliders from `20240530LGSK02024`. Seven
accessible fastball/slider clips were inspected from this game; one was excluded
because the landing foot is occluded. The other 31 player clips have no event
audit. This is a convenience sample selected to exercise the interface, not a
representative estimate of this player or season.

`data/mechanics/event_audit.json` stores inclusive, zero-based local-frame
brackets. The coding agent inspected consecutive **original crop images** for
the descending lead forefoot approaching sustained support, and the transition
from hand/ball overlap to a visibly separated ball. The brackets deliberately
include repeated and ambiguous images. They are provisional image-event
interpretations; **no independent human review, force-based contact measurement,
or validated ball detector is claimed**. The public pose quality review remains
unchanged. Source images are not published.

The throwing side comes from the historical player profile: right-handed means
left lead leg; left-handed means right lead leg. Unknown sides are excluded.
The current six clips share one game and a center-field live broadcast view,
with differing zoom and framing. No cross-camera calibration is claimed.

## Calculation

The source is the original Halpe-26 crop-pixel pose, not a resampled animation.
Left lead leg uses indices 11–13–15; right uses 12–14–16.

`projected flexion = 180° − acos((hip−knee)·(ankle−knee) / (|hip−knee||ankle−knee|))`

Zero is a straight leg **in the image plane**. It is not anatomical sagittal
knee flexion. Camera projection and occlusion can strongly change the pattern.

The exploratory screen requires all three model scores ≥0.4, finite nondegenerate
segments, and no interpolated/outlier joints throughout the full audited range.
These scores are not calibrated probabilities. The separate strict screen uses
≥0.5 and currently retains only one pitch. The strict screen does not provide
aggregate summaries with fewer than three pitches.

For each pitch, the landing and release bracket midpoints define normalized
progress 0–100%. The original local/source frame indices and broadcast-playback
duration interval remain available. A 41-point waveform linearly resamples the
supported interval. These are not 41 independent observations. No missing source
frames are filled. Users can select raw, 3-frame or 5-frame centered box means;
filtering is restricted to the fully supported audited span.

Reported quantities are:

| Field | Definition |
| --- | --- |
| `landingFlexion` | Projected flexion at landing-bracket midpoint |
| `extraFlexion` | Maximum flexion within the aligned interval minus landing flexion |
| `extension` | Maximum flexion minus release flexion; positive means straightening from maximum bend |
| `netExtension` | Landing flexion minus release flexion; this distinct quantity is retained in JSON |
| `maximumFlexionProgress` | First maximum-flexion frame as % of the aligned interval; **not peak extension velocity** |

Each pitch has one weight. For each pitch type, ≥3 pitches permit a pointwise
median/IQR waveform and scalar median/IQR. The band describes **across-pitch
variation**, mixing genuine variation, projection effects and estimation error;
it is not a model confidence interval. A selected curve always comes from an
actual pitch, and all distribution/scatter points open that pitch's evidence.
The speed chart does not fit a regression or infer causality for three pitches.

## Sensitivity and limits

For each pitch, 27 alternatives combine landing start/midpoint/end, release
start/midpoint/end and raw/3/5-frame means. Their scalar min–max envelope is
reported separately from across-pitch IQR. It is an operational sensitivity
analysis, not a confidence interval or a model accuracy bound.

An optional local image-cadence audit compares adjacent grayscale 160×160 crops.
Mean absolute differences <1 on a 0–255 scale flag near-duplicate candidates.
Roughly alternating near-duplicates occur in these 59.94-fps files. This does
not certify the true sampling rate or slow-motion factor. Broadcast elapsed
seconds are labeled as playback time. No physical milliseconds, angular
velocity, maximum extension speed or intersegment peak-time offset is reported.

Current native 3D endpoint flexion is retained in `modelInspection`, explicitly
ineligible for aggregation. Its change differs markedly from the 2D projection
in these clips. Different representations are not interchangeable, and that
comparison validates neither representation's anatomical accuracy. Cross-model
stability, camera calibration and independent event/pose validation remain
pending. Pelvis–trunk separation and rotation timing are deferred until their
coordinate, event and clock prerequisites are met. There is no force, braking,
energy-transfer efficiency, injury-risk or overall mechanics grade.

Definitions are informed by the [lead-knee study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11329978/)
and [in-game markerless study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11544755/),
without adopting their laboratory/system-specific reference values. The
[OpenCap Monocular best-practice page](https://www.opencap.ai/best-practices?variant=monocular)
does not validate this broadcast-pitching reconstruction workflow.

## Build, inspect and share

```bash
# Uses public poses, event brackets and the checked-in cadence audit; stdlib only.
python3 scripts/build_mechanics_report.py
# Optional: recompute image-cadence diagnostics from unpublished local source crops.
python3 scripts/build_mechanics_report.py --audit-video
python3 -m unittest discover -s tests -p test_mechanics_report.py
python3 tests/mechanics_browser_qa.py http://127.0.0.1:8000 /path/to/chrome
```

`data/mechanics/65320.json` includes original-pose/model SHA-256 values, bracket
source frames, scalar definitions, all three processing settings, sensitivity,
source-record/broadcast links, screen results and the exhaustive exclusion list.
Canonical clips, reconstructions, source poses and linkage audits are not edited.

The report is reached through the player's delivery tab or
`?pitcher=65320&mechanics=lead-knee#pitchers`. Sharing preserves pitch type, view,
selected actual pitch, phase progress, observation screen and processing choice
in URL parameters. Other players display their pending event-audit status and
an entry to the pilot; their unaudited clips are not assigned invented events.
