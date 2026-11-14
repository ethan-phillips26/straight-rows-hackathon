# Straight Rows

Bluestaq × NDSU Straight Rows hackathon entry. Detects degraded-GNSS-guidance episodes on a simulated
3-tractor fleet (Casselton, ND; 72 h of 1 Hz telemetry plus simulated SDA feeds). Output per episode:
start, end, tractors, cause class, action, confidence, evidence. Detections cover the unlabeled
evaluation window (2026-05-06T00:00Z to 2026-05-07T00:00Z).

Results are retrospective: detectors use the whole evaluation window at once, so onsets are not
real-time reportable. Repo: `<repo url>` · GitHub: `<handle>`

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
python scripts/run_all.py --data starter/data --out submission.json --github-handle <handle> --repo-url <repo url>
python starter/submission/validate_submission.py submission.json    # expect: OK
```

Deterministic: no RNG, sorted outputs, no clock dependence. Two runs produce identical files.

```bash
pytest -q                                        # unit tests (synthetic ones need no dataset)
python scripts/eval_dev.py --detections X.json   # score against the 12 labeled dev episodes
python scripts/eval_dev.py --baseline            # organizers' baseline on dev
python scripts/verify_findings.py                # prints every number cited under "Data findings"
python scripts/make_viz.py --events labels --pick TRN-03,TRN-06,TRN-05   # replay viewer
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

<!-- Update to match the final code. -->

1. `features.py`: 1 s features per tractor; baseline from that tractor's own quiet working time (median/MAD);
   same signals relative to the fleet median (separates regional from local causes).
2. `detect.py`: one independent detector per class, so a storm does not mask a jammer or machine fault.
3. `attribute.py`: rule-based class assignment. Space-side feeds (Kp, Dst, S4, advisories, RFI) support or
   rule out alternatives. Rules are kept to one-sentence each; 12 labeled episodes do not support a larger model.
4. `output.py`: writes and validates `submission.json`; fails if detections > 495 (validator limit is 500).

**Multipath.** Scoring credits one representative multipath episode and exempts other recurrent ones from
penalty unless they duplicate it. Dev shows the representative is an ordinary pass (not the first, tractor
varies). Strategy: one single-tractor detection per tractor per north-headland pass (~486/day), driven by the
pass geometry and 355 s cadence rather than the dip, so passes masked by jamming or a storm are still
emitted. Evidence notes whether the other tractors dipped simultaneously. Risk: depends on the scorer's
exemption wording; this is the least certain design decision.

**Use of labels.** The 12 dev episodes (hours 0-48) were used to characterise signatures, set thresholds, and
score with the organizers' matcher. No tuning on the evaluation window. Its unlabeled data is used for
baselines (permitted).

## Results

<!-- TODO: fill from scripts/eval_dev.py. -->

Dev (12 episodes), organizers' diagnostic (max 40; ignores confidence and evidence):

| | diagnostic | macro-F1 | onset credit |
|---|---|---|---|
| organizers' baseline | 20.15 | 0.508 | 0.497 |
| this entry | TODO | TODO | TODO |

TODO: per-class precision/recall, onset error per episode, ablation by feed, threshold sensitivity.

## Confidence

<!-- TODO: document the actual mapping and a dev reliability table. -->

Derived from the number of independent agreeing signals and how fully alternatives are ruled out; reduced
when evidence conflicts (e.g. jamming signature during an active storm). Dev has 12 episodes, so any
calibration estimate has wide error.

## Failure modes

<!-- TODO: update from dev errors. -->

- Onset of slow ramps (spoofing XTE leaves the noise floor gradually): start is back-extrapolated, error of minutes possible.
- Simultaneous onset of two causes (e.g. jammer and storm): fleet-wide subtraction cannot separate them reliably.
- A real single-tractor headland dip is indistinguishable from a benign multipath pass.
- RFI feed misses events and lags onset by 10-15 min; absence of a report is treated as no information.
- Thresholds set on 12 episodes; likely overfit to them.

## Extra data that would help

- Independent position source (INS, wheel odometry) for direct spoofing detection instead of XTE.
- RFI reports with accurate timestamps and geolocation.
- Per-satellite C/N0 (multipath affects 1-2 satellites, jamming all).
- More labeled episodes, especially concurrent ones, for calibration.
- Real (non-synthetic) farm telemetry.

## Replay viewer

`scripts/make_viz.py` builds a standalone HTML replay: tractor motion, signal charts, raw or residual
(vs own baseline / vs fleet median, ±3σ MAD band), and the detection's class, action, confidence, evidence.

```bash
python scripts/make_viz.py --events labels --pick TRN-03,TRN-06,TRN-05
python scripts/make_viz.py --events submission.json --pick 1,2 --name final
```

Output: `viz/out/<name>.html` (offline). Embeds dataset telemetry; gitignored, not part of the submission.

## Layout

```
starter/        organizers' starter (fetcher, baseline, validator), unmodified
straightrows/   load, features, detect, attribute, output
scripts/        run_all.py, eval_dev.py, verify_findings.py, make_viz.py
scoring/        organizers' scorer, unmodified
tests/          pytest
viz/            replay viewer template
```
