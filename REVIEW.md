# Code review and upgrade notes

Reviewed all supplied Python scripts, frontend files, configuration, workflow, README and generated JSON structure. Updated September 14, 2026.

## Correctness fixes

| Finding | Change |
| --- | --- |
| Player historical rates were divided by 90 twice | Keep historical and positional fallback rates in consistent per-minute units |
| Player stats were joined using rounded floats | Join calculated statistics by preserved row index |
| Squad selection could choose several goalkeepers and unavailable players | Search plausible formations, require one goalkeeper, exclude unavailable statuses; allow incomplete bench |
| Fixed ten-goal probability grid lost tail mass | Adaptive Poisson grid with normalized probabilities |
| Season simulations ignored calibrated match odds | Sample scores from an outcome-calibrated grid |
| Each exact tie received full probability at multiple positions | Allocate equal fractional mass across tied positions |
| Home and away priors used one pooled scoring average | Shrink each rate to the corresponding venue scoring average |
| Cross-division normalization used only the target league, making it ineffective | Remove misleading normalization and apply explicitly heuristic stronger shrinkage |
| Distant forecasts artificially weakened current ratings through time decay | Freeze forward forecasts to today's information date |
| Today's availability and squad projection were repeated for months | Restrict player projections to a 14-day horizon |
| Archive years and prior age were hard-coded | Derive years from the date and imported history |
| Archive numeric suffixes and accents broke player matching | Normalize both, while documenting remaining identity ambiguity |
| Empty imports failed obscurely; duplicates could overweight matches | Clear empty-source error and duplicate fixture removal |

## Design and performance

A new cream, forest-green and lime dashboard replaces the narrow plain layout. It includes responsive match cards, competition and gameweek navigation, a season view, team search across gameweeks, summary metrics and an expandable methodology panel. Text retains numerical probabilities so bars are not the only way to interpret a forecast.

Accessibility provisions include a skip link, visible keyboard focus, labelled search/navigation, pressed button states, native details elements, table header scopes and reduced-motion support. These are implementation provisions, not a certified accessibility audit.

Repeated match calculations are reused for season simulations. Season rankings use NumPy sorting, with Python loops only for exact ties. Squad table markup is deferred until a match opens. JSON is compact, and player forecasts are no longer generated for the entire remaining season. The bundled payload is much smaller principally because invalid legacy squad/season results were removed; this is not a like-for-like performance benchmark.

The frontend now checks HTTP errors, provides retry, tolerates an unavailable colour or evaluation file, validates colour strings and escapes names. Missing forecasts and stale data have explicit states.

## Verification

- 13 Python tests pass, including a full isolated synthetic CSV → evaluation → forecast → season JSON build.
- Regression coverage includes high-goal probability tails, zero-goal histories, calibration consistency, exact ties, same-day leakage protection, legal formations, unavailable players, historical units, rounded-stat joins and parser formats.
- JavaScript syntax and lightweight UI logic checks pass for loading, search, empty states, ordinal formatting, escaping and archived-snapshot disclosure.
- GitHub Actions runs both Python and JavaScript checks before importing source data and deploying.
- Real browser screenshots, responsive layout and interaction checks **could not run**: no browser binary was installed, and the browser download was denied by the environment's network policy. The responsive styling therefore needs a real browser review.
- The full historical rebuild and before/after accuracy comparison **could not run**: raw training CSVs were absent from the ZIP and the source repository download returned HTTP 403 under the environment's network policy.
- GitHub Pages deployment itself has not been run from this workspace.

The included preview retains the original match forecast date and identifies those forecasts as version 1 archived estimates. Known-affected player and season outputs are withheld. The original evaluation is explicitly labelled as belonging to the original model. The code generates version 2 outputs when rebuilt; no improved accuracy numbers are claimed.

## Next modelling experiments

These require training data and chronological comparisons before adoption:

1. Fit opponent-adjusted attack/defence strengths rather than directly averaging scores. Compare both log loss and calibration against this corrected baseline.
2. Fit a low-score dependency correction and tune time decay on earlier validation windows. Do not hard-code a draw boost without measuring it.
3. Learn promotion/relegation transition effects from clubs moving between divisions. Scoring averages alone do not measure competition strength.
4. Use fixture-level recent minutes, confirmed injury dates, stable player identifiers and observed substitutions to fit expected minutes and start probabilities. Handle missing historical xG coverage explicitly.
5. Add dynamic team-strength uncertainty to season simulations, plus schedule completeness checks, points deductions and competition-specific head-to-head handling.
6. Add rolling-origin evaluation across multiple seasons and per-league baselines, with uncertainty intervals. Reserve a final untouched period for model selection decisions.

The current corrections remove known implementation errors and improve transparency. They do not establish that the forecasts beat either the previous model or other forecasting systems.
