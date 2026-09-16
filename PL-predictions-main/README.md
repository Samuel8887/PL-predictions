# Touchline — English Football Forecasts

A dependency-free, responsive dashboard for Premier League, Championship and League One match probabilities, player projections and season simulations.

## Preview the included snapshot

```sh
python -m http.server 8000 --directory site
```

Open http://localhost:8000. The included September 10, 2026 match forecasts are **archived outputs from the original model**, clearly labelled in the dashboard. Original player and season projections were withheld because the audit found calculation errors in those outputs. They return after rebuilding with the corrected model. An empty League One view reflects the supplied dataset; fixtures have not been invented.

## Rebuild with current data

Requires Python 3.10+ and Git, plus access to GitHub, raw.githubusercontent.com and fantasy.premierleague.com.

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/import_data.py
python scripts/import_player_data.py
python scripts/train_predict.py
python -m http.server 8000 --directory site
```

The match importer clones openfootball when its source directory is absent. If reusing a checkout, update that checkout before importing; the importer does not silently overwrite a user's checkout. `--source path/to/england` selects a local source. `--as-of YYYY-MM-DD` limits match-result availability, but **does not provide historical player snapshots**: do not use the current-player importer for a historical player backtest.

GitHub Pages: choose **Settings → Pages → GitHub Actions**, then push to `main` or run the workflow manually. The daily build installs dependencies, runs regression tests, imports data and generates the static site. The workflow caches Python packages.

## What's changed

- New Touchline dashboard: responsive cards, competition navigation, all-gameweek team search, summary metrics, method disclosure, explicit loading failures and stale-data notices.
- Squad tables render only when a match is opened. Unavailable season forecasts have an explicit empty state.
- A Poisson score grid now extends far enough to retain essentially all probability mass, instead of always stopping at ten goals.
- Season score sampling uses the calibrated outcome probabilities shown on match cards. Score ratios within each result class are preserved.
- Exact table ties share probability correctly. Each team's finishing distribution sums to one before display rounding. Ranking is vectorized; only exact ties require individual handling.
- Forecasts are reused between fixture generation and season simulations. Future forecasts use today's information date rather than artificially aging ratings toward the prior as the fixture date recedes.
- Home and away shrinkage use their respective scoring baselines. Sparse cross-division histories receive stronger shrinkage; the previous division adjustment was ineffective.
- Player priors use consistent per-minute units. Player statistics join by row identity instead of rounded floating-point values, which previously caused valid projections to become zero.
- Eligible starting XIs must fit a plausible formation with exactly one goalkeeper. Known unavailable statuses are excluded instead of being reintroduced to fill a bench. If a full formation is impossible, no complete XI is claimed.
- Player goal/assist allocation weights estimated playing time. Current availability is used only for fixtures within 14 days; it is not extended across the full season.
- Player history dates are dynamic. Matching normalizes accents and archive numeric suffixes. Fixtures are deduplicated, and an empty import fails clearly.
- Compact JSON and no extra frontend framework or asset dependencies.

## Model and interpretation

The team model remains an interpretable baseline: exponentially weighted home/away scoring and conceding rates, a two-year half-life, an eight-year lookback and shrinkage toward league averages. A geometric blend combines each attack with the opposing defence. Six equivalent prior games are used normally and eighteen for cross-division fallback. The latter is a conservative heuristic, **not a fitted promotion-strength adjustment**.

Outcome calibration selects an exponent using earlier chronological validation fixtures. The final chronological 20% is sampled deterministically for evaluation, with at most 600 fixtures. Same-day and future results are excluded from every historical prediction. Log loss, multiclass Brier score, accuracy and confidence bins are reported against a pooled historical-outcome baseline.

Expected goals on the cards are the **underlying Poisson rates**. Calibrating the outcome classes changes the resulting score distribution's mean; these displayed rates are not the post-calibration simulation means. Season outlooks include expected points and an 80% points interval in JSON. The fixed seed makes a build reproducible for identical inputs.

Season ranking uses points, goal difference and goals scored, sharing exact ties evenly. It does not implement head-to-head or playoff rules, points deductions, or uncertainty about future team-strength changes. It requires forecasts for every remaining imported fixture, but cannot know whether the upstream schedule is complete.

Player start chances, minutes and assist allocation are heuristics, not empirically calibrated probabilities. Season totals do not capture recent substitutions, tactical formations or upcoming rotation fully. Injuries and availability depend on the source snapshot. Historical name matching can still miss renamed players or collide for namesakes; stable cross-season player IDs would be preferable. Older missing xG/xA coverage also limits the historical prior.

## Verification and remaining work

See `REVIEW.md` for the audit, verification results and suggested next modelling experiments. Offline regression tests cover the known calculation failures and a complete synthetic build. They establish implementation correctness for those cases, **not improved prediction accuracy**. New held-out performance must be measured after importing the source history.

For numerical details, see [SciPy's Poisson distribution documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html).

## Files

- `scripts/import_data.py`: Football.TXT match parser and importer.
- `scripts/import_player_data.py`: historical/current FPL player importer.
- `scripts/train_predict.py`: model, calibration, evaluation and season simulations.
- `tests/`: offline model and build regression tests.
- `config/`: team aliases and display colours.
- `site/`: static frontend and included archived preview.
- `.github/workflows/deploy.yml`: tested daily build and Pages deployment.

Match source: [openfootball/england](https://github.com/openfootball/england), public domain. Player sources: Fantasy Premier League and [vaastav's archive](https://github.com/vaastav/Fantasy-Premier-League). This is independent educational analysis, not betting advice.
