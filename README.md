# KBO Pitch Lab

The [Explorer](https://kbo-pitch-db.github.io/#explore) shows final RTMPose Halpe-26 and one published GVHMR + ground/contact IK + MotionBERT reconstruction per clip. The current canonical index contains **813 clips / 123 pitchers / 135,741 source frames**. Every clip uses `gvhmr-ground-motionbert-v1`; there are **zero WHAM fallbacks**. No pitcher is selected automatically. `?clip=CLIP_ID#explore` opens a clip directly.

Release: `gvhmr-ground-motionbert-20261028-v1`. All **813** clips use actual MotionBERT inference and both-arm refinement, fixed native body shape/segment lengths, exact inferred contact anchors and sequence-fixed rigid shoes. This replaces every default 3D payload, including the formerly targeted Choi Won-tae slider. All original 2D JSON/NPZ bytes and source timestamps are preserved. Eight source-inspected pitcher-to-batter tail transitions have separately versioned RTMPose recoveries; their normal prefix frames remain exactly unchanged. The other 805 active final-2D paths are unchanged from the pre-combination release. Earlier GVHMR/contact and arm-study results remain frozen as historical comparisons; they are not the current default.

All clips use the same viewer, one clip-wide camera/scale and one inferred ground frame. No framewise recentering, endpoint hold, population-stride target or display trajectory correction is applied. Playback stops at the last frame. JSON exports the saved coordinates; 3D CSV exports 24 native SMPL joint centers, source timing and availability. COCO-17 surface landmarks and six foot markers remain supplementary data. Rigid saved shoes also retain their shape between rendered frames; interpolation preserves a marker that is stationary in both saved frames.

## Player reports, v2 (2026-10-28)

The English/Korean website now uses focused routes and three report tabs: **At a glance → Pitches & results → Delivery**. Opening a player replaces the directory; research metadata and advanced count/sequence tables are disclosed on demand. Language and report filters persist during language changes. Official player names remain as recorded.

**All acquired records are included.** The 32 cached games contain 9,754 distinct pitches by **155 pitchers**, including 806 pitches by 32 players with no collected motion. The original motion sample remains **813 clips / 123 players**. Reports start from every acquired pitch, not just automatically linked highlights. Neither the records nor the clips are a complete KBO season. `?pitcher=65320#pitchers` opens a report; `?clip=CLIP_ID#explore` opens the full 2D/3D viewer.

The overview combines a plain-language reading of the pitch mix, an explicitly defined repertoire type, raw pitch-based statistics and a five-axis characteristic plot with percentile bars. The visual structure is informed by [Baseball Savant player reports](https://baseballsavant.mlb.com/savant-player/paul-skenes-694973); the data and reference population here are KBO acquired-game records, not MLB Statcast or full-season league rankings. No synthetic overall grade is calculated. A larger radar shape is not uniformly better, particularly for zone rate and breaking-ball share.

- **Statistics:** fastball average speed uses records labeled `직구`. Whiff rate is `S/(S+F+H)`; called-plus-swinging-strike rate is `(T+S)/(B+T+S+F+H)`. Naver's text confirms `T=called strike`, `S=swinging strike`, `F=foul`, `H=in play`, `B=ball`, `V=missed bunt`, `W=foul bunt`. Bunt attempts are excluded from the ordinary-swing and CSW denominators. No strikeout rate or ERA is inferred from pitch outcome codes.
- **Relative positions:** midrank percentiles are computed independently for each metric among pitchers with ≥50 acquired pitches. Additional thresholds are 15 fastballs for velocity, 20 non-bunt swings for whiff rate and 50 supported locations for zone rate. A reference group must have ≥5 players. Eligibility, metric denominators, reference-group sizes and medians are exported; small samples are not assigned ranks. These are descriptive sample comparisons, not calibrated talent estimates.
- **Types:** fastball/two-seam/cutter share ≥55% → fastball-led; slider/curve share ≥45% → breaking-ball-led; changeup/forkball share ≥30% → offspeed-led; otherwise mixed. Fewer than 30 pitches is labeled limited sample. The rule and caveat are visible in the report. It is not a learned scouting classification.
- **Readable repertoire:** usage bars preserve the full record denominator; velocity ranges show the mean, middle 50% and observed range with their units; a 5×5 location-density grid outlines the recorded strike zone. Filters update the related views together. Users can switch to individual points, select a region, inspect translated outcomes and follow actual record or motion links. Count and within-at-bat sequence tables remain available under a disclosure.
- **Linkage:** 390 automatically accepted links / 33 held / 390 unresolved, zero human-validated links. A join still requires accepted audit evidence and agreement on game, player identity and pitch ID. A record without an accepted motion link retains its statistics and source URL. Different clips of one pitch share the same record denominator; the current accepted population has no duplicate replays.
- **Reference delivery:** when the most-used pitch has ≥2 collected clips, compare those; otherwise use the most-collected available pitch type. Sample the four supported 2D angle curves at 21 normalized clip positions, require ≥42 supported components and choose the actual clip closest to the componentwise median with a missing-support penalty. This is an example representative of the captured views, not an averaged skeleton or validated event-aligned template. A sparse fallback is labeled an example only. The player presents a real numerical sequence, three actual snapshots and a linked full 2D/3D viewer; no other player's motion is substituted.

### Mechanics and display

The guide explains what to watch, why a joint angle helps, what 180° means, and the limitations of the coordinates. It shows one readable curve at a time, synchronized with the actual delivery. **3D model included angles** are the default when a time-aligned reconstruction is available; users can switch to **2D image angles** and unsmoothed observations. Native model angles are continuous model estimates, not independent physical observations or validated anatomical rotations. They are not Baseball Savant's shoulder-to-ball [arm angle](https://baseballsavant.mlb.com/leaderboard/pitcher-arm-angles).

Both modes use `acos((a−b)·(c−b)/(|a−b||c−b|))`, in degrees:

| Joint | Halpe-26 image triplet | SMPL-24 model triplet |
| --- | --- | --- |
| Left knee | 11–13–15 | 1–4–7 |
| Right knee | 12–14–16 | 2–5–8 |
| Left elbow | 5–7–9 | 16–18–20 |
| Right elbow | 6–8–10 | 17–19–21 |

2D requires all three confidences ≥0.5 and excludes interpolated/outlier joints and degenerate segments. 3D requires three finite native joint centers and excludes `available=false` frames. Model timing must match the final 2D sequence. Native-model gaps, including camera-cut exclusions, are never bridged.

**Display processing never changes source/downloaded values.** The readable curve applies a five-frame weighted mean within supported runs. In 2D only, internal gaps may be linearly bridged when they span at most 3 missing frames and 100 ms, with endpoint change ≤45°. These segments are dashed; longer/unsupported gaps are shaded and retained. Raw mode disables smoothing and display bridges. Bounded cubic display segments avoid overshoot between values. Current-frame readouts always use the unmodified source/model angle. `pitch_charts.js` owns these display-only transforms. Curve axes resize with their container so labels remain legible on mobile.

Comparison uses actual 2D sequences with a shared normalized-progress slider and native 3D curves when both reconstructions exist; otherwise both curves use 2D image angles. Original source timing is retained separately for each clip. This is not foot-plant or release alignment. No landing/release duration, physical angular velocity, force-transfer grade, MVPI, HB/IVB or spin estimate is derived from these poses.

PTS x is `cross_plate_x_ft`; z evaluates the provider's quadratic trajectory at `cross_plate_y_ft`, both in feet. Density cells use `(z−bottom_sz)/(top_sz−bottom_sz)` with the central zone width 17 inches. Only supported PTS records and ordered finite zone bounds enter location statistics. Source-model consistency does not establish independent tracking accuracy; no target location is available, so location is not labeled command.

### Rebuild and verify

```bash
python3 scripts/build_portal_data.py
python3 -m unittest discover -s tests -p test_portal_data.py
node tests/portal_math.test.cjs
# With Playwright Python and Chromium installed, against a static local server:
python3 tests/portal_browser_qa.py http://127.0.0.1:8000 /path/to/chrome
```

The generator reads cached inputs and writes deterministic schema-v2 metadata; it does not alter canonical clips, poses or audit decisions. `data/portal/summary.json` includes source hashes, all 155 profiles and reference-group definitions. Player JSON includes all that player's acquired records, joins, statistics, reference delivery and observation counts. Portal data URLs have their own version, independently of the pose release, preventing older cached summaries from breaking a new UI. Schema/release mismatches fail explicitly. The complete `data/portal/pitch_records.csv.gz` preserves the source CSV; `raw_file` entries refer to local cached provenance rather than hosted raw responses. [Example code](docs/portal-example.py) follows an accepted pitch link to a final 2D sequence. Broadcast footage, model files and private research inputs are not packaged by this workflow. Third-party source/model terms remain separate from the site's metadata.

## What the reconstruction means

The official GVHMR learned checkpoint runs with final RTMPose and HMR2a pitcher-crop inputs (with full-image bbox coordinates), 30 Hz network resampling and identity camera-angular-velocity hypotheses. Its shared SMPL-X rotations are approximately adapted to SMPL, with a separate WHAM-derived **SMPL shape only** for the original inputs. Versioned subject recoveries instead use the corrected pitcher’s HMR2a RGB shape. This is **not** the complete native SMPL-X GVHMR demo/postprocessor. Camera, pose and root are then fitted in stages; the contact step jointly refines lower body, trunk, root and a shared focal estimate, preserving fixed sequence shape and local upper-limb rotations.

The new contact reconstruction uses independent image-supported toe episodes, including clips without a detected plant event. It optimizes a clip-wide ground plane/camera and fixed-shape native joint rotations/root. Bounded two-link leg IK reconstructs actual saved rotations at supporting ankles; it does not stretch leg bones or clamp feet in the renderer. A final rigid-shoe pose fit jointly constrains rotation and ankle-trajectory continuity and uses the original reliable 2D foot observations, native leg reach and one shared plane. Shoe marker offsets are constant in each ankle frame, avoiding the variable toe–heel distance of skinned surface markers. Inferred support can release and recontact; unsupported moving feet remain free.

Actual MotionBERT inference uses the separately acquired pretrained model plus the existing public-mocap last-block adaptation. Original and flip-restored predictions remain separate depth hypotheses. Both arms are reconstructed from the source elbow/wrist rays and fixed limb lengths, with MotionBERT conditioning depth selection and temporal refinement. No claim is made that direct MotionBERT transplantation improves every arm or identifies unique depth. Per-clip checkpoint/prediction/solver hashes and selected refinements are included in the saved JSON.

All 813 outputs must pass checks on saved coordinates: inferred contact-marker diameter ≤1% of the frozen model-leg length, native FK/segment consistency, rigid shoe lengths, bounded knees, all-foot/native-body floor penetration, root/ankle velocity changes, shoe rotation continuity and the closest shoe remaining within 8% of the modeled leg length above the inferred plane. For 66 population clips and one recovered tail, actual root/foot/camera parameters are jointly refitted to prevent whole-body flight outside static toe episodes; this is an explicit pitching-support prior, not a measured contact detector. The absolute modeled root-speed diagnostic remains visible but does not reject continuous fast motion: neither broadcast slow-motion factor nor physical scale is calibrated. Contact residual reduction is objective-linked, **not independent evidence of correct KBO 3D motion**.

Strict source checks are retained separately: overall median, wrist p95 and foot p95 reprojection must each increase by no more than 2 px to pass. **38/813** pass all strict source checks. Foot p95 regresses by over 2 px in **774** clips, overall median in **90**, and wrist p95 in **0**. Rigid contact/shoe constraints trade away image agreement, so physical consistency is not a claim of uniformly improved accuracy. Every failed source check, before/after error and tail remains in the per-clip quality JSON and [combined release audit](data/combined_release_audit.csv). A geometry pass never relabels a failed source check. [The combined summary](data/combined_release_summary.json) records the actual population counts. No clip is labeled human manually reviewed, benchmark eligible or validated for absolute stride. Camera, depth, player scale and actual contact remain monocular estimates.

Six source clips have confirmed camera changes at their ends. Their 14 post-cut frames preserve final 2D/timestamps but retain null 3D coordinates, `available=false` and `unavailable_reason=source_camera_change`. There are **135,727 valid numerical 3D frames** out of 135,741 source frames. Playback and CSV preserve these gaps.

[Current model summary](data/motion_summary.json) and [frozen execution/input/selection provenance](experiments/pitch_observation/releases/combined-20261028/) describe the actual method. A single constant scene transform and inferred ground plane apply to every valid frame. The viewer does not invent a world trajectory, population stride or measured mound. Source time is broadcast playback time and is not converted into physical angular velocities.

## Run locally

Serve only this repository directory: `python3 -m http.server 8000`, then open `http://localhost:8000`. Keep the parent workspace, source broadcast media, model weights and licensed body files outside the served directory.

The site is static GitHub Pages. Only skeleton media and numerical poses are hosted. Timestamped YouTube links and optional click-to-load embeds open source footage; no broadcast pixels/audio are packaged. Research comparisons are unlisted direct links with `noindex,nofollow,noarchive`, absent from the main navigation and disallowed in `robots.txt`; they are not access-controlled.

All 813 clips retain player identity evidence and all 123 pitchers retain 2024 KBO portraits with attribution. Identity evidence and 3D motion quality are separate. Estimates are not validated biomechanics or force data.

## Reproduce, resume and verify

Use the existing separate research worktree and prepared Python/CUDA environment. Original local `dataset/`, separately acquired official model weights, the public-mocap adaptation and licensed neutral SMPL are required; weights/body files/source broadcast media are excluded from Pages. Frozen inputs, intermediate selection and solver source hashes are recorded in the release provenance. `common.py` resolves paths from the research worktree; copying scripts into an unrelated folder changes that root.

```bash
PY=../env/bin/python
EXP=experiments/pitch_observation
$PY $EXP/combined_population.py prepare
$PY $EXP/combined_population.py motionbert
$PY $EXP/combined_support.py
$PY $EXP/combined_arms.py --once
$PY $EXP/combined_contact_ik.py --once
# Reproduce the versioned repairs and selection_frozen.json recorded in provenance.
$PY $EXP/combined_foot_pose_temporal.py
$PY $EXP/audit_combined_temporal.py --once
$PY $EXP/repair_combined_temporal.py --once
$PY $EXP/repair_root_support.py
$PY $EXP/recover_subject_tails.py observe
$PY $EXP/recover_subject_tails.py features
$PY $EXP/recover_subject_tails.py network
$PY $EXP/recover_subject_tails.py base
$PY $EXP/recover_subject_tails.py combine
$PY $EXP/package_subject_tails.py
$PY $EXP/adopt_subject_tails.py
$PY $EXP/combined_subject_update.py
$PY $EXP/adopt_combined_subject.py
$PY $EXP/select_combined_release.py
$PY $EXP/package_combined_population.py
# After saved-coordinate and staged payload checks:
$PY $EXP/package_combined_population.py --promote
$PY -m unittest discover -s tests
node tests/motion_viewer.test.cjs
node tests/reconstruction_registry.test.cjs
node tests/combined_population_browser_qa.cjs http://127.0.0.1:8000/
# Read-only all-813 live index/pose/quality/final-2D content verification:
$PY $EXP/verify_live_combined.py
```

Completed per-clip NPZ/JSON/hashes remain resumable in private `combined_population_v1/`. Source and old published estimates are immutable. Generation/auditing/staging do not push or deploy. Failed candidates remain in private versioned folders; no fallback is silently substituted for a required combined output. Native FK and serialized contact/rigid-shoe checks are integrity checks, not held-out 3D accuracy validation.

## Preserved earlier results

The original WHAM index is losslessly archived in `data/releases/wham-20261028/original_index.json.gz`. `raw_pose_archive.json` and `scene_pose_archive.json` map old raw/scene paths to lossless gzip files and record decoded SHA-256 and original Git blob IDs. Historical scene JSON is losslessly compressed to keep the static site within its storage budget. The pre-combination canonical index is preserved byte-for-byte in `data/releases/support-20261028/original_index.json.gz`, and earlier GVHMR poses/Git history remain intact. Original final 2D files were not overwritten; the eight corrected tails use new versioned paths and preserve every earlier good frame. The [archived README](docs/archive/README_wham_20261028.md) documents the earlier WHAM workflow and its historical 44/769 screening counts; those labels are not transferred to this new method.

## Wrong-person input correction

Oh Seung-hwan Delivery 3 (`4e09dd724a8c_c0011`) originally tracked the batter from the 2D bbox onward. Its active 2D and GVHMR reconstruction now use a separately versioned pitcher track on the same 75 source frames; all original final 2D and 3D files are preserved. The original batter's WHAM pose and shape do not initialize the replacement. Its corrected input is retained in the combined release. Eight additional tail-only input recoveries are listed in `data/combined_release_summary.json`; their old 2D/3D assets remain available. The interval starts after stride: leg lift/landing are outside the observation. The historical subject-correction fit had unknown contact; the new combined release uses separately inferred image-supported toe episodes without inventing the missing stride interval. This is a subject-selection correction, not validation of absolute stride or 3D accuracy. See the [recovery commands and evidence](docs/subject_corrections/oh-delivery3.md) and [versioned asset hashes](data/subject_corrections/4e09dd724a8c_c0011-v1.json).

## Separate arm-cocking experiment

After verifying the full main release, three versioned arm experiments were executed on six tuning and six player/source-disjoint confirmation clips. That historical experiment did not change the then-current release. The unlisted `studies/arm-cocking-20261028/` contains synchronized fixed-axis comparisons, all 16 V3 candidates, the three-iteration report, CSV/JSON metrics, frozen selection, model provenance and `reproduce.txt`. It is absent from the homepage navigation and is not access-controlled.

Direct MotionBERT arm transplantation and additional fitting can worsen source correspondence. The final preview instead intersects source elbow/wrist rays with fixed-length limbs, retaining alternative depth branches; MotionBERT supplies a relative-depth prior. Confirmation wrist p95 median is 5.54 → 1.62 px (frames excluded from this arm refinement: 4.96 → 2.82 px). A GV-depth control also obtains similar pixel gains, and an independent RTMW observation provides little corroboration. These results do not establish correct arm depth, wrist flexion, biomechanical accuracy or a unique solution. Yang Hyeon-jong's source-fit failure and Choi Won-tae's reduced posterior-hand proxy are retained as counterexamples. Native wrist rotations, body/root/camera and feet remain fixed by experimental scope; their preservation is not new contact evidence. The frozen twelve-clip study is preserved as-is. The current all-813 combined release runs its own actual per-clip inference, arm/contact refinement and audits; it does not relabel the old study as a population validation.
