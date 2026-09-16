# Validation and fixes — 16 September 2026

This review supersedes the earlier offline-only audit. The downloads, real-data build and Chromium checks succeeded. No public deployment was performed.

## Changes implemented

- Fixed round-heading variants, carried-forward kickoff times, postponed/awarded annotations, rescheduled fixture duplication, playoff contamination and COVID-delayed summer dates. Regular-season team identities no longer contain score or status text.
- Supplemented incomplete current data with official FPL fixtures and football-data.co.uk results. Added explicit schedule audits; incomplete schedules and unresolved past results cannot silently produce season tables.
- Replaced player-name matching with persistent FPL codes across 2016–17 through 2025–26. All 7,358 player-season rows map to codes; 528 of 659 current players have historical coverage. Remaining players use positional priors. All current team names match the Premier League schedule.
- Corrected missing-xG exposure denominators, guarded zero-rate allocations, capped starting probabilities by positional capacity, and bounded scoring/assisting chances by availability. Same-day/future results and unfinished rows cannot enter historical team predictions.
- Added fixed seasonal evaluation with an original-code comparison and a per-league historical baseline. No speculative parameter upgrades were selected using test results.
- Fixed controls during loading, retry behavior, access to fixtures with unassigned rounds, keyboard focus in expandable cards, keyboard-scrollable tables and mobile scroll hints. Removed a transient low-contrast league-button animation exposed by repeated accessibility checks.
- Added dependency constraints, real-data/artifact validators, reproducible browser checks and a workflow that actually runs the checks before producing a Pages artifact.

## Real chronological evaluation

Validation: 2024-07-01 through 2025-07-01 (exclusive); 482 eligible sampled fixtures. Both implementations selected exponent **1.0**. The final test covers **2025-08-01 through 2026-05-24**, all **1,484** regular-season fixtures in 2025–26; **0 skipped**.

Original code: Git revision `45ad019`. Both implementations received the same repaired match histories and identical fixture-date information cutoffs. The comparison isolates the team-probability implementation; it does not compare historical player projections or reproduce the old corrupt input parser. Final outcomes did not choose parameters. A discovered date-parsing bug was corrected and the same evaluation rerun without tuning.

Brier score is the sum across the three outcome classes (range 0–2). ECE is top-choice confidence calibration error using 0.2-wide bins; smaller is better, but a constant baseline can have low ECE without useful discrimination.

| League / model | N | Log loss | Brier | Accuracy | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| All leagues — corrected | 1484 | 1.05822 | 0.63801 | 45.62% | 0.02016 |
| All leagues — original | 1484 | 1.05911 | 0.63863 | 45.82% | 0.02470 |
| All leagues — baseline | 1484 | 1.07570 | 0.65088 | 43.53% | 0.00507 |
| Premier League — corrected | 380 | 1.04305 | 0.62634 | 48.16% | 0.04128 |
| Premier League — original | 380 | 1.04356 | 0.62665 | 48.68% | 0.03627 |
| Premier League — baseline | 380 | 1.08183 | 0.65508 | 42.63% | 0.03154 |
| Championship — corrected | 552 | 1.07318 | 0.64920 | 42.75% | 0.01507 |
| Championship — original | 552 | 1.07312 | 0.64910 | 42.57% | 0.01650 |
| Championship — baseline | 552 | 1.08285 | 0.65598 | 41.67% | 0.01792 |
| League One — corrected | 552 | 1.05371 | 0.63484 | 46.74% | 0.04086 |
| League One — original | 552 | 1.05581 | 0.63642 | 47.10% | 0.04520 |
| League One — baseline | 552 | 1.06432 | 0.64291 | 46.01% | 0.02599 |

Per-league test dates: Premier League **15 August 2025–24 May 2026**; Championship **8 August 2025–2 May 2026**; League One **1 August 2025–2 May 2026**. The baseline uses only each league’s pre-test outcomes, with add-one smoothing. Later test predictions can incorporate earlier test results, always excluding the fixture day and future results.

The corrected model slightly improves aggregate log loss and Brier score but reduces accuracy versus the original. The descriptive paired date-cluster bootstrap estimates corrected-minus-original log loss at **−0.000886**, with a 95% interval of approximately **[−0.002045, +0.000192]**. This interval crosses zero. There is **no convincing evidence of an overall accuracy improvement** over the original. The corrected model outperforms the constant historical baseline on aggregate log loss, Brier and accuracy in this one season; that is not evidence of general superiority.

Full confidence bins, all evaluated fixture probabilities and dates are in [evaluation.json](site/data/evaluation.json) and [evaluation_fixtures.json](site/data/evaluation_fixtures.json). The reproducible descriptive uncertainty calculation is [compare_uncertainty.py](scripts/compare_uncertainty.py). No historical player accuracy claim is made.

## Forecast refresh and coverage

- **Premier League:** 380/380 scheduled matches, 40 confirmed results and 340 upcoming fixtures. A 20-team, 10,000-run season outlook and 20 projected squads across ten near-term matches were generated.
- **Championship:** 552/552 scheduled matches, 81 confirmed results, 469 upcoming fixtures and two unresolved past results: Bristol City–Lincoln City and Middlesbrough–Millwall, both 15 September. The available feeds had not supplied confirmed scores; its season table is withheld.
- **League One:** 72/552 fixtures available for 2026–27: 71 confirmed results and AFC Wimbledon–Milton Keynes Dons on 17 September. The openfootball repository has no current-season League One file. The secondary source publishes completed results and a short forthcoming-fixture feed, not a full season schedule. This repairs the completely empty view but cannot establish season completeness.
- The archived preview was replaced only after the real-data build passed output invariants. Fresh JSON includes matches, near-term player projections, evaluation, Premier League season outlook, source coverage and player-identity reports. Source URLs, hashes and snapshot timestamps are recorded.

## Tests and actual browser verification

- **23 Python regression tests pass.** They cover normalization, calibrated score sampling, tied finishing distributions, formations, identity aliases, missing-xG exposure, zero-rate allocation, availability caps, parser variants, COVID dates, duplicate schedules, chronological leakage and an isolated build.
- JavaScript syntax and frontend logic tests pass. `pip check` reports no broken requirements. The complete generated artifact passes strict JSON and forecast invariants. The real-data audit verifies fixture/player identity uniqueness, dates, current availability and published player probabilities.
- Chromium **153.0.8010.12** tested the real rebuilt site under `/PL-predictions/` at **1440×1000, 834×1112 and 390×844**. League switching, previous/next gameweeks, cross-week team search, no results, squads, repeated expansion, season tables and missing season tables passed.
- Separate fault-injection scenarios passed for delayed loading, HTTP failure/retry, missing optional files, stale data, a missing league and an unassigned gameweek. Intentional fault responses were isolated from normal-request checks.
- Keyboard skip navigation, summary activation, visible focus and horizontal table scrolling passed. No page-level horizontal overflow, normal-page console errors or failed asset requests were observed. Axe-core found **zero violations** against WCAG A/AA checks in the match, expanded-squad and season-table states at all three widths. Screenshots were visually inspected. Axe marked the arrow-only gameweek control for manual contrast review; its enabled ink/white color pair measures 14.65:1. This is not a screen-reader certification or a test on physical mobile hardware.
- Squad tables were absent before expansion, created only for the opened fixture, and not duplicated by repeated toggles. The forecast payload is about **361 KB**. Local unthrottled measurements follow; they are not real-device/network benchmarks.

| Viewport | Initial load | Search interaction |
| --- | ---: | ---: |
| 1440×1000 | 179 ms | 19 ms |
| 834×1112 | 103 ms | 18 ms |
| 390×844 | 107 ms | 18 ms |

Screenshots: [desktop](artifacts/browser/desktop.png), [tablet](artifacts/browser/tablet.png), [390px mobile](artifacts/browser/mobile.png), [mobile squad](artifacts/browser/mobile-squad.png), [mobile season table](artifacts/browser/mobile-season.png). Machine-readable evidence is in [browser results](artifacts/browser/results.json) and [source audit](artifacts/source_audit.json).

## Deployment and remaining limits

The workflow passed **actionlint 1.7.12**. Its Python, JavaScript, import, generation, artifact-validation and Chromium commands were exercised locally. It uses a full Git checkout so the original implementation is available, caches package downloads, uploads verification evidence, and prevents pull-request deployment. A hosted Linux GitHub Actions execution and the actual public Pages deployment remain **untested**; neither was triggered.

The model does not account for deductions, head-to-head standings rules, promotion playoffs, future transfers, changing team strength or schedule uncertainty. Cross-division shrinkage and all player probabilities remain heuristics. Today’s player snapshot was not used as historical evidence. Browser testing used Chromium only; Firefox, Safari, physical touch devices and a full assistive-technology audit remain untested.

Exact installation, live rebuild, offline replay, audit and browser commands are in [README.md](README.md). No model parameters were changed in response to final-test performance.
