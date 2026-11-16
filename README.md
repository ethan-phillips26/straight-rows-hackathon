# Straight Rows

My entry to the Bluestaq × NDSU "Straight Rows" hackathon (24 hours, solo, 14–15 Nov 2026).
**[Try the data explorer](https://ethanphillips.dev/straight-rows-hackathon/)**

## The hackathon

Three tractors work a field near Casselton, ND, steered by GNSS. The organizers released 72 hours of synthetic 1 Hz
telemetry (position, cross-track error, C/N0, satellites, fix state, base-station link) plus simulated Space Domain
Awareness (SDA) feeds: Kp, Dst, S4 scintillation, RFI reports and advisories.

The task: in an **unlabeled evaluation day** (2026-05-06T00:00Z to 2026-05-07T00:00Z, 8 hidden episodes), find every
episode of degraded guidance and report its start, end, affected tractors, cause class, recommended action, confidence
and checkable evidence. The first 48 hours come with 12 labeled "dev" episodes to learn from.

| Cause | Recommended action |
|---|---|
| SPACE_WEATHER | HALT_OR_DEGRADED_MODE |
| JAMMING | HALT_AND_REPORT |
| SPOOFING | HALT_CROSSCHECK_AND_REPORT |
| EQUIPMENT_FAULT | SERVICE_MACHINE_OTHERS_CONTINUE |
| MULTIPATH | CONTINUE_FALSE_ALARM |

The hard part is that causes overlap (a jammer inside a storm, a spoof inside an antenna fault) and "all three tractors
affected" does not prove space weather. Judges score a 100-point rubric (technical execution 30, mission relevance and
decision quality 25, research depth 20, assurance and rigor 15, demo 10). An automated diagnostic worth up to 40 is
supporting evidence only.

## My implementation

One detector per cause, each working on **residuals** (a tractor's signal minus the fleet median, or minus its own quiet
baseline) so a fleet-wide effect like a storm doesn't hide a local one. Every rule is physical and short enough to
explain in one sentence; 12 labeled episodes don't support a bigger model.

| Cause | What the data does | How it is detected |
|---|---|---|
| SPACE_WEATHER | All 3 tractors step together, in line with S4 (corr -0.998 over storm slots) | Fleet-wide C/N0 drop that sits on the S4 line. S4 alone isn't enough: a quiet-farm S4 bump on 05-05 did nothing |
| JAMMING | C/N0 falls with distance to one fixed source (~20 dB/decade); nearer tractors are hit harder | C/N0 drop not explained by S4. The source is located from a C/N0 fit and its position and error ellipse are cited |
| SPOOFING | One tractor; C/N0, sats and fix look normal, but reported position drifts | Reported lon offset from the lockstep line, which must be backed by that tractor's own cross-track error |
| EQUIPMENT_FAULT | One tractor, neighbours clean: antenna dropouts or `base_link_ok` false about 30% of the time | Per-tractor min-C/N0, satellite and link residuals |
| MULTIPATH | A 17 s min-C/N0 dip at every north-headland pass (972/972 dev passes) | One short single-tractor detection per pass, driven by the pass geometry, so passes inside a storm or jam are still found |

Code (`straightrows/`):
- `features.py` builds a 1 Hz table, a 1-minute table (headland seconds excluded, worst-case columns kept) and a per-pass table.
- `detect.py` runs the independent detectors. Episodes are clipped to the evaluation window.
- `evidence.py` writes the evidence strings. Every number is measured from the data, never typed in, and each line says what rules out the alternatives (for example "S4 quiet, so not space weather").
- `attribute.py` maps class to action, sets confidence and attaches evidence.
- `output.py` writes and validates `submission.json`.

**Confidence** is set by expert rules, not fitted. Each class starts from a base (0.80–0.90) and is lowered by named
conditions, for example jamming inside a storm (-0.10) or a spoof whose cross-track error doesn't show the offset
(-0.40). The last evidence line of each detection lists the adjustments.

**Use of labels.** The 12 dev episodes were used to characterise signatures, set thresholds, and score with the
organizers' matcher. Nothing was tuned on the evaluation window. Its unlabeled data was used only for baselines.

**Data explorer.** `make explorer` builds an offline HTML viewer of all 72 hours: events as lanes, a top-down field map
with the tractors, signal charts, SDA feeds, playback, and each detection's evidence. The hosted copy is published from
`site/` by GitHub Pages (`make site`).

## Results

Organizers' automated diagnostic on the evaluation day (max 40; scored 2026-11-14 22:40 UTC, published in the organizer
repo's `results/results.json`):

| | score | macro-F1 | onset credit | episodes matched | action agreement | false positives |
|---|---|---|---|---|---|---|
| **this entry** | **40 / 40** | 1.00 | 1.00 | 8 / 8 | 8 / 8 | 0 |

The submission is valid and on time: 493 detections (7 cause episodes plus 486 short multipath passes). 9 of the 16
submissions scored 40/40, so the diagnostic does not separate the top entries. The listed rank (6) is only an order
within that tie. **The judges' 100-point rubric score is not published**, so I don't have it.

On the dev days, the starter baseline scores 20.15 and this entry scores 40.00. That dev number is **in-sample**: the
rules were read from those 12 episodes. Checks that go beyond it:
- `scripts/loo_episodes.py` re-derives each threshold without one episode of its class: all 10 non-multipath episodes are still detected, 0 false positives. The detector structure was designed having seen all 12, so this tests robustness, not unseen data.
- `scripts/sweep_thresholds.py` moves each constant by x0.8 and x1.2: the dev result holds for every constant except the storm line, which holds within about ±3.75 dB against a measured residual of 0.53 dB.
- An independent review of the detectors found variants that dev never showed. The three tests in `tests/test_review_unseen_variants.py` fail on purpose: each is a known open limitation.

## Failure modes and limits

- Slow ramps (a spoof's cross-track error leaves the noise floor gradually): the start is back-extrapolated and can be off by minutes.
- Two causes starting at the same time (for example a jammer and a storm) can't be reliably separated by fleet-wide subtraction.
- A real single-tractor headland dip looks the same as a benign multipath pass.
- The RFI feed misses events and lags onsets by 10–15 minutes, so a missing report counts as no information.
- Events shorter than the minimum durations are not reported: jamming under 24 min, spoofing under 10 min, equipment under 18 min.
- All three tractors spoofed by the same offset preserves the lockstep formation, so nothing is detected.
- Thresholds were set on 12 episodes and are probably overfit to them.
- Results are retrospective: detectors use the whole evaluation window at once, so onsets are not real-time reportable.

**What extra data would help:** an independent position source (INS or wheel odometry) for direct spoofing detection,
accurate RFI timestamps and geolocation, per-satellite C/N0 (multipath hits one or two satellites, jamming all of them),
more labeled and concurrent episodes for calibration, and real farm telemetry.

## Setup and run

Python 3.11+ (developed on 3.14.9). Dependencies are pinned in `requirements.txt`. No credentials and no network at run
time; the only download is the dataset, which is SHA-256 verified.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python starter/scripts/get_data.py          # downloads the organizers' zip to starter/data/

# regenerate submission.json and validate it (expect: OK)
make submission HANDLE=ethan-phillips26 REPO=https://github.com/ethan-phillips26/straight-rows-hackathon

make test        # pytest (synthetic tests need no dataset)
make dev         # score against the 12 labeled dev episodes
make explorer    # build the viewer -> viz/out/explorer.html
make site        # explorer + copy to site/index.html for GitHub Pages
```

## Layout

```
starter/        organizers' starter (fetcher, baseline, validator), unmodified
straightrows/   load, features, detect, attribute, evidence, output
scripts/        run_all.py, eval_dev.py, loo_episodes.py, sweep_thresholds.py, verify_findings.py, make_explorer.py
scoring/        organizers' scorer, unmodified
tests/          pytest
viz/            explorer_template.html
site/           the built explorer published by GitHub Pages
```
