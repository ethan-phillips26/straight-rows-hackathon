# Straight Rows

Bluestaq × NDSU Straight Rows hackathon entry. Detects degraded-GNSS-guidance episodes on a simulated
3-tractor fleet (Casselton, ND; 72 h of 1 Hz telemetry plus simulated SDA feeds). Output per episode:
start, end, tractors, cause class, action, confidence, evidence. Detections cover the unlabeled
evaluation window (2026-05-06T00:00Z to 2026-05-07T00:00Z).

Results are retrospective: detectors use the whole evaluation window at once, so onsets are not
real-time reportable. Repo: `https://github.com/ethan-phillips26/straight-rows-hackathon` · GitHub: `ethan-phillips26`

## Setup and run

Python 3.11+ (developed on 3.14.9). Dependencies pinned in `requirements.txt` (numpy 2.3.3, pandas 2.3.3,
matplotlib 3.10.6 = starter pins; pytest 9.2.0, jupyterlab 4.6.6 added). No credentials. The only network
use is the one-time dataset download.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python starter/scripts/get_data.py          # downloads organizers' zip, verifies SHA-256, extracts to starter/data/
```

Regenerate `submission.json`:

```bash
python scripts/run_all.py --data starter/data --out submission.json --github-handle ethan-phillips26 --repo-url https://github.com/ethan-phillips26/straight-rows-hackathon
# same thing plus validation: make submission HANDLE=ethan-phillips26 REPO=https://github.com/ethan-phillips26/straight-rows-hackathon   (also: make test, make dev, make sweep)
python starter/submission/validate_submission.py submission.json    # expect: OK
```

Deterministic: no RNG, sorted outputs, no clock dependence. Two runs produce identical files.

```bash
pytest -q                                        # unit tests (synthetic ones need no dataset)
python scripts/eval_dev.py --detections X.json   # score against the 12 labeled dev episodes
python scripts/eval_dev.py --baseline            # organizers' baseline on dev
python scripts/verify_findings.py                # prints every number cited under "Data findings"
make explorer                                    # current viewer -> viz/out/explorer.html
```

## Data findings (dev, hours 0-48; reproduced by `scripts/verify_findings.py`)

| Class | Signature |
|---|---|
| Baseline | Tractors in lockstep, 4.5 m/s. Quiet working time: zero seconds with fix != RTK_FIXED, link down, sats < 11, mean C/N0 < 42.9 dB-Hz, or \|XTE\| > 0.45 m |
| SPACE_WEATHER | Step on all 3 tractors, aligned with an S4 step. corr(S4, fleet C/N0) = -0.998 over storm slots. S4 ~0.3 for 2.5 h on 05-05 had no farm impact, so S4 alone is insufficient |
| JAMMING | C/N0 follows distance to one fixed source, ~20 dB/decade (fit b = 19-23). Nearer tractors hit harder; oscillation = driving toward/away. TRN-04: farthest tractor unlabeled |
| SPOOFING | One tractor. C/N0, sats, fix unchanged. Reported lon offset equals XTE (max abs diff <= 0.33 m, 3/3). Storm/jamming do not move reported position (offset <= 0.05 m) |
| EQUIPMENT_FAULT | One tractor, neighbours clean. Antenna: scattered min-C/N0 and sat dropouts. Link: `base_link_ok` false ~30% of working time |
| MULTIPATH | 17 s min-C/N0 dip at every north-headland pass on all tractors (972/972 dev, 486/486 eval). Labels match the dip to the second |

Overlaps in dev: jamming inside a storm (TRN-05), spoofing inside a storm (TRN-08), spoofing inside an
antenna fault on the same tractor (TRN-06 in TRN-09), multipath inside a storm (TRN-12). Each cause is
detected on residuals after removing the fleet-wide effect and reported as a separate detection.

## Approach

1. `features.py`: builds three tables from the telemetry and SDA feeds (`build_features`). A 1 Hz table has each tractor's
   value minus the fleet median (separates regional from local causes), its C/N0 drop versus its own quiet baseline,
   and `lon_dev_m`, the reported position's offset from the lockstep formation line (the spoofing signal). A 1-minute
   table, with the north-headland seconds excluded, serves the slow causes (storm, jamming, spoofing, equipment) and
   keeps worst-case columns so short dropouts are not averaged away. S4, Kp, Dst and the advisory flag are joined
   as-of. A per-pass table (one row per tractor per north-headland pass) feeds multipath, since the 17 s dip would
   vanish at 1-minute resolution. No labels or tuned thresholds are used.
2. `detect.py`: one independent detector per class, so a storm does not mask a jammer or machine fault. A "storm" run
   must also look like one (all three tractors alike, fleet on the S4 line); otherwise the drop is left to the jamming
   detector. A spoof's lon offset must be backed by that tractor's own cross-track error, or the tractors whose XTE ramps
   are blamed instead. Episodes are clipped to the evaluation window, not filtered on their start.
3. `evidence.py`: writes the checkable strings. Every number is measured from the data (never typed in), and overlapping
   causes are compared against neighbours, so a spoof inside a storm does not claim a "normal" receiver.
4. `attribute.py`: class -> action, confidence (see below) and evidence. Space-side feeds (Kp, Dst, S4, advisories, RFI)
   support or rule out alternatives. Rules are kept to one-sentence each; 12 labeled episodes do not support a larger model.
5. `output.py`: writes and validates `submission.json`; fails if detections > 500 (the validator's limit; eval emits 493).

**Multipath.** Scoring credits one representative multipath episode and exempts other recurrent ones from
penalty unless they duplicate it. Dev shows the representative is an ordinary pass (not the first, tractor
varies). Strategy: one single-tractor detection per tractor per north-headland pass (~486/day), driven by the
pass geometry and 355 s cadence rather than the dip, so passes masked by jamming or a storm are still
emitted. Evidence notes whether the other tractors dipped simultaneously. Risk: depends on the scorer's
exemption wording; this is the least certain design decision.

**Use of labels.** The 12 dev episodes (hours 0-48) were used to characterise signatures, set thresholds, and
score with the organizers' matcher. No tuning on the evaluation window. Its unlabeled data is used for
baselines (permitted). The detectors were also run on the evaluation day to look at the shapes of what they find
(local notes, not in this repo); no threshold or label was derived from that look.

## Results

Dev (12 episodes), organizers' diagnostic (max 40; ignores confidence and evidence):

| | diagnostic | macro-F1 | onset credit |
|---|---|---|---|
| organizers' baseline | 20.15 | 0.508 | 0.497 |
| this entry | 40.00 | 1.000 | 1.000 |

**This is not a held-out score.** The rules were read from these same 12 episodes, so 40/40 shows the rules describe them,
not that they generalise. A leave-one-episode-out check (`python scripts/loo_episodes.py`) re-derives each gap constant
without one episode of its class: all 10 non-multipath episodes are still detected strictly, 0 false positives. Only the
numbers are re-derived; the detector structure was designed having seen all 12, so this is a robustness check, not unseen-data
validation. An independent review found no rule that looks likely to cost an evaluation episode, and found four plausible
variants dev never showed (fixed; see Out of scope).
Per-class precision/recall and onset errors: `python scripts/run_dev.py`.

**Threshold sensitivity** (`python scripts/sweep_thresholds.py`, each constant x0.8 and x1.2, strict criterion: right class,
exact tractor set, onset within 5 min, no false positive). All but one constant leave dev at 40.00 and strict-ok at both
ends. The exception is the storm line `STORM_CN0_AT_S4_0` (42.7 dB): a percentage is the wrong unit, because 20 % is
8.5 dB. In dB the dev result holds from -3.75 to +3.75 dB, against a measured residual of at most 0.53 dB on the three
storm episodes (so about 6x). Earlier notes said "every threshold swept x0.8/x1.2: no change"; that was true for the
diagnostic score only and is superseded by this paragraph.

## Confidence

Expert-set, not fitted (`attribute.py`): each class starts from a base (storm 0.90, jamming 0.85, spoofing 0.90,
equipment 0.80, multipath 0.90) and is lowered by named conditions: for example a jamming tractor within 2 dB of the
4.35 dB cut-off (-0.10), jamming inside a storm (-0.10), a spoof whose cross-track error does not show the offset (-0.40),
a multipath pass where not all three tractors dipped (-0.30). The only increase is +0.05 when an RFI report within an hour
agrees with a jamming call. The last evidence line of every detection lists its adjustments. There is no reliability table:
12 dev episodes cannot support one, and the dev result is in-sample.

## Failure modes

- Onset of slow ramps (spoofing XTE leaves the noise floor gradually): start is back-extrapolated, error of minutes possible.
- Simultaneous onset of two causes (e.g. jammer and storm): fleet-wide subtraction cannot separate them reliably.
- A real single-tractor headland dip is indistinguishable from a benign multipath pass.
- RFI feed misses events and lags onset by 10-15 min; absence of a report is treated as no information.
- Thresholds set on 12 episodes; likely overfit to them.

**Out of scope / known limits** (from the independent review):
- Events shorter than the minimum durations are not reported: jamming < 24 min, spoofing < 10 min, equipment < 18 min.
  An 18-minute jam was verified to give no detection.
- A link fault active in only ~4 % of working seconds is found about 18 minutes late (at 10 % it is exact).
- All three tractors spoofed by the same offset: the lockstep formation is preserved, so nothing is detected. Two spoofed
  by the same offset are found through their cross-track error, with the lockstep offset only locating the episode.
- A uniform three-tractor jam that happens to sit exactly on the S4-predicted storm line cannot be told from a storm
  with farm data alone.
- The storm line (C/N0 = 42.7 - 17.7 x S4) is fitted on two storms; its tolerance is about +-3.75 dB (see Results).
- Lines to be ready to explain: the lockstep mode picking in `features.py`, the run-splitting idiom in `find_runs`, and the
  600 s cap on the spoof onset back-extrapolation (comments in the code say why).

## Extra data that would help

- Independent position source (INS, wheel odometry) for direct spoofing detection instead of XTE.
- RFI reports with accurate timestamps and geolocation.
- Per-satellite C/N0 (multipath affects 1-2 satellites, jamming all).
- More labeled episodes, especially concurrent ones, for calibration.
- Real (non-synthetic) farm telemetry.

## Data explorer (current viewer)

`scripts/make_explorer.py` (or `make explorer`) builds the viewer to use: the whole 72 h of 1 Hz telemetry and SDA feeds in one
offline HTML file, with dev labels and our detections as lanes. Click an event to replay it: top-down field with the tractors
(cross-track error shown as a smoothed east-west bar, jammer fit when available), signal charts raw or as residuals, and the
detection's class, action, confidence and evidence.

```bash
make explorer            # == python scripts/make_explorer.py --data starter/data --detections submission.json
```

Output: `viz/out/explorer.html` (offline, gitignored, not part of the submission).

Past viewer: `scripts/make_viz.py` builds per-event replay clips (`viz/out/<name>.html`). It still runs but is no longer
maintained or used for the demo; the explorer replaces it.

## Layout

```
starter/        organizers' starter (fetcher, baseline, validator), unmodified
straightrows/   load, features, detect, attribute, output
scripts/        run_all.py, eval_dev.py, verify_findings.py, loo_episodes.py, make_explorer.py (current viewer), make_viz.py (past viewer),
                multipath_sidereal.py, xte_spoof_check.py (analysis)
scoring/        organizers' scorer, unmodified
tests/          pytest
viz/            explorer_template.html (current), viewer_template.html (past)
```
