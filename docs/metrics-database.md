# Pitch mechanics database

Calculation release: `mechanics-db-20261028-v1`. Pose release remains
`gvhmr-ground-motionbert-20261028-v1`. Source poses, reconstruction files, pitch
links and the earlier event pilot are preserved.

The primary route is `#database`; an empty hash opens it. Player pages open a
metric summary by default. Existing repertoire/results, descriptive player
overview, phase pilot and synchronized individual-pitch viewers remain in
their separate tabs/details. There is no automatic mechanics efficiency grade.

## Units and scope

| View | Row unit | Rows |
| --- | --- | ---: |
| Linked motion (default) | Unique provider pitch record with an accepted automatic motion link | 390 |
| All acquired records | Unique provider pitch record, including pitches without motion | 9,754 |
| Unlinked motion | Unresolved/held motion clip; not a verified unique pitch | 423 |

There are 813 motion clips associated with 123 players. The current 390 accepted
links are all to distinct pitch records; the schema nevertheless retains every
alternate clip and specifies a representative selection rule to prevent future
replays inflating pitch counts. Unlinked clips have `pitchId:null`. Their game
is sourced from the independently date/team-matched source-video catalog; it
does **not** imply acceptance of a particular pitch-record link. Their record
URL remains null.

One whole observation interval is selected per record: prefer an existing
provisional event-audited clip, otherwise the clip with the largest sum of
supported 2D metric frame counts, with deterministic ties. A row never mixes
different replay windows for different metrics. All alternate clip IDs remain
available. Automatic links are not independently human validated.

The complete database has 10,177 rows, containing the two explicitly separated
units. It is not a complete season. Statistics never treat those combined rows
as 10,177 verified physical pitches.

## Metric dictionary and calculation

`data/metrics/dictionary.json` defines every metric's ID, bilingual name and
definition, unit, coordinate system, reference event/window, required joints,
formula, calculation version, conditions and limitations. The same dictionary
drives table groups, definitions, axis selectors and export units.

There are 24 defined columns:

| Family | Columns | Current computation |
| --- | ---: | --- |
| Original 2D observation statistics | 8 | Lead-knee and throwing-elbow projected flexion P10/P90/range; trunk image-vertical tilt P90; upper-arm/downward-trunk projected angle P90 |
| Native 3D inspection geometry | 6 | Horizontal shoulder–pelvis separation P50/P90, observed pelvis yaw range, vertical trunk inclination, upper-arm/downward-trunk angle, leg-normalized horizontal ankle separation |
| Provisional event-aligned 2D | 5 | Landing knee flexion, extra bending, straightening after maximum bend, net straightening from landing, maximum-bend phase |
| Dependency-gated reserved columns | 5 | Physical landing–release time, pelvis–trunk peak offset, release forward/lateral trunk lean, normalized landing stride |

The 8 event-free 2D statistics describe the **entire observed clip**, which can
contain preparation and recovery. They are not landing–release measurements,
and clip boundary, camera and replay differences matter. The source is the
original published Halpe-26 observation, not interpolated display animation.

Projected flexion is `180° − included angle`; 0° is projected straight. Role
mapping uses recorded throwing handedness: R → left lead knee/right throwing
elbow; L → right lead knee/left throwing elbow. Unknown roles produce nulls.
P10/P90 use linearly interpolated sample quantiles. The observed range is
P90−P10, not maximum excursion. These statistics weight supported decoded frames;
duplicated broadcast frames are not certified independent physical samples.

Required original 2D joints need model scores ≥0.5, finite nondegenerate
geometry, no interpolated/outlier flags, and at least 10 supported frames and
50% of the clip. A failed elbow gate does not remove a valid knee statistic.
No missing frame is filled.

Native SMPL-24 geometry uses the published clip-fixed inferred frame with Y up.
Source frame/time alignment must agree with the original 2D. Required original
2D observations also pass the joint gates. Almost-vertical shoulder/hip lines
are excluded when their horizontal length is ≤20% of their 3D length. Pelvis
yaw is unwrapped only within the longest contiguous supported run; no gaps are
bridged. The normalized ankle distance is dimensionless, using ankle XZ
separation divided by mean hip–knee–ankle segment length. It is **not stride**.
Trunk inclination is against estimated vertical, **not forward or lateral lean
in a calibrated pitching frame**. Shoulder-line separation is not anatomical
thorax–pelvis rotation.

All numeric 3D inspection cells are `needs_review_3d`. They are stored and
visible for inspection, but excluded from default summaries and charts. An
explicit opt-in includes them and is recorded in the analysis URL and aggregate
CSV. These values have not passed independent anatomical or cross-model
validation, and no laboratory reference ranges or normative player types are
assigned. The earlier pilot's 3D aggregation restriction remains the default.

The five event columns reuse the existing [lead-knee pilot](mechanics-pilot.md)
without expanding its six provisional brackets to other clips. They preserve
`lead-knee-pilot-20261028-v1` calculation version, ≥0.4 original-score screen,
three-frame box mean, midpoint event alignment, strict-screen availability and
separate scalar sensitivity envelopes. Their status is `provisional_event`.
They are distinct from the event-free ≥0.5 columns, not silently comparable
versions of one definition.

Physical-clock columns are null because repeated frames and unknown slow
motion have not been calibrated into physical time. Forward/lateral release
lean and landing stride are null because the pitching reference frame is not
validated. Those values are never approximated using clip progress, total clip
duration, ankle separation or another similarly named statistic.

## Missingness and summaries

Each available observation row has a metric map. Each cell contains `value`
(number or null), `status`, `validFrames`, `totalFrames`, and a frame range where
applicable. No-motion records store `metrics:null`, a schema-defined shorthand
for every dictionary cell having `value:null`, `status:no_motion`, and zero
valid/total frame **counts**. The numeric metric value is never filled with zero.
CSV expands the shorthand into empty numeric cells and explicit statuses.

Statuses separate observable 2D statistics, provisional events, review-required
3D, missing motion, unaudited events, insufficient observations, unknown
throwing side, unverified physical time, unverified pitching frame and model
alignment mismatch. Source quality flags remain on the row and in provenance;
having a finite value is not independent validation.

Summaries group player × pitch type within the selected scope, games and source
videos. Each row contributes one weight. Every metric has its own eligible n,
median and IQR. IQR is shown only for n≥3; the median is retained for n≥1 with n
visible. An IQR is across-row variation, not model/event uncertainty. Unlinked
scope summarizes **clips**, with explicit unit labels. All-record summaries can
have many source pitches but only a small valid metric n.

Sorting uses the selected column and always places nulls last in both
directions. Filtering includes player, pitch type, game, source video, name/ID,
and selected-metric status. Source video filtering does not certify matching
camera angle. Tables paginate at 20 rows, while CSV exports every filtered row.
Cell distributions and scatter points open the corresponding record and source
clip. Scatter axes use values from the same row/observation interval; no causal
or efficiency interpretation is generated.

## Files and reproduction

| Artifact | Contents |
| --- | --- |
| `database.json` / lossless `database.json.gz` | Scope, dictionary, status codes, rows, per-metric availability, representative policy, input hashes |
| `dictionary.json` | Standalone versioned dictionary and status labels |
| `clip_provenance.json` | Per-clip original/model paths, SHA-256, frame count/FPS, quality flags and model release |
| `pitches.csv.gz` | All 10,177 rows with expanded per-metric value/unit/status/frame counts/version |

The browser prefers the 344 KB compressed database with native decompression
and falls back to plain JSON. UI CSV includes selected metric columns and
their unit, status, observation counts, event/window, version and source clip.
Full and filtered pitch CSV use identical column names. Aggregate CSV includes selected medians/IQR/n, per-metric source-status counts and the 3D inclusion flag. UTF-8
BOM supports Korean spreadsheet import. Missing metric cells stay blank.

```bash
python3 scripts/build_metrics_database.py
python3 -m unittest discover -s tests -p test_metrics_database.py
node tests/metrics_database.test.cjs
python3 tests/metrics_database_browser_qa.py http://127.0.0.1:8000 /path/to/chrome
```

The builder reads published clips/poses, player records and event-pilot files,
plus the local cached `../dataset/sources.csv` game map. Its digest is retained.
Outputs and gzip timestamps are deterministic. No original broadcast images,
new source media, model weights, credentials or private raw responses are
published. Source and model terms remain separate.
