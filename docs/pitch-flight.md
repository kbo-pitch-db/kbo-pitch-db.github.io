# Naver PTS pitch-flight release

Version: `naver-pts-flight-20261029-v1`.

The input is the existing `dataset/naver_sports/pitches.csv`, collected from the public Naver Sports game relay JSON for the 32 acquired 2024 games. Raw relay `ptsOptions` supplies initial position (`x0/y0/z0`), velocity (`vx0/vy0/vz0`) and acceleration (`ax/ay/az`). Pitch IDs join the existing game/player records; video availability does not enter the denominator. Of 9,754 distinct pitches, 9,747 have finite, supported trajectories. Seven lack tracking and retain null HB, IVB and VAA.

These are **derived trajectory values**, not directly reported Naver or Statcast HB/IVB fields. Naver's undocumented coordinate interpretation follows the existing PITCHf/x fit, with the supplied crossing x independently checked against the trajectory to within 0.05 ft by the collector. This consistency check is not independent physical calibration. No spin rate, spin axis or force-transfer score is inferred.

## Definition

Movement is calculated from y=40 ft to the front edge of home plate at y=17/12 ft. Positive x is catcher's right; positive z is up. Values use cm. VAA is in degrees, negative downward, evaluated at the plate front. These values should not be silently compared with release-to-plate movement or measurements using a different reference or correction.

For each axis, position follows `r(t) = r0 + v0*t + a*t²/2`. Solve the y equation for the forward-time roots at the start and plate, requiring velocity toward the plate and a supported interval.

Following Alan M. Nathan's mean-velocity drag correction, let `g=(0,0,-32.174)` ft/s², `a_ng=a-g`, and let `v_hat` be the unit mean velocity over the interval. Remove the acceleration parallel to flight:

```
a_induced = a_ng - dot(a_ng, v_hat) * v_hat
HB_cm     = 0.5 * a_induced.x * dt² * 30.48
IVB_cm    = 0.5 * a_induced.z * dt² * 30.48
VAA_deg   = degrees(atan2(vz0 + az*t_plate, -(vy0 + ay*t_plate)))
```

The approximation removes both gravity and the mean drag component. It is not the older `0.5*(az+g)*dt²` approximation, which retains vertical drag. It does not establish the underlying PTS sensor's calibration or precision. All missing/invalid fits stay null.

## Data and checks

- `data/portal/pitch_flight.json(.gz)`: per-pitch values, status, input hash, reference and dictionary.
- `data/portal/pitch_flight.csv.gz`: values, IDs, source URL, version and the nine input coefficients.
- `scripts/build_pitch_flight.py`: deterministic standard-library generator.
- `tests/test_pitch_flight.py`: gravity-only and pure-drag controls, horizontal reflection, analytic VAA, missingness and complete pitch-ID coverage.
- `tests/pitch_flight_browser_qa.py`: real charts, shared pitch selection, exact row values, complete exports, grouping, missing physical timing, old shares, languages and mobile layouts.

Player charts show pitch-type means and metric-specific valid n, without zero imputation. Flight summaries pool by actual player/type; they do not split by pose model or filming conditions. Biomechanics retains its original grouping and acceptance rules. No original pose or audit file is changed.

## References

- [Alan M. Nathan, Determining the Movement of a Pitched Baseball (2012)](https://baseball.physics.illinois.edu/Movement.pdf), especially the appendix and configurable 40 ft reference.
- [Naver Sports KBO](https://m.sports.naver.com/baseball/kbo/schedule/index); individual relay source URLs accompany the records.
- [User-provided KBO data example](https://github.com/leekichang/where-do-the-robot-umpires-see). This supplied context for revisiting the original relay data; no code or data from that repository is republished here.
