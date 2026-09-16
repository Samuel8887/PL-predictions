# Touchline — English Football Forecasts

A static, responsive dashboard for Premier League, Championship and League One match probabilities, projected squads and season simulations. The frontend has no runtime package dependencies.

## Verified snapshot

The bundled forecasts were rebuilt with real downloaded data on **16 September 2026**, using model **2.1**. They replace the archived preview. There are **340 Premier League, 469 Championship and 1 League One upcoming fixtures**, with player projections for the ten Premier League fixtures in the next 14 days. The Premier League season outlook uses 10,000 simulations.

Coverage is explicit: Championship has two past fixtures without confirmed results, so its season table is withheld. The main source has no 2026–27 League One file; the supplemental source supplies 71 results and one forthcoming fixture, not a full schedule. Its season table is also withheld. No missing fixtures or results have been invented.

See [REVIEW.md](REVIEW.md) for the real holdout comparison, limitations and verification evidence. See [browser screenshots and results](artifacts/browser/results.json) and [source audit](artifacts/source_audit.json).

## Preview locally

```sh
python -m http.server 8000 --directory site
```

Open http://localhost:8000. Asset and data paths are relative and have also been tested under `/PL-predictions/`, the repository Pages subpath.

## Install and rebuild

Use Python 3.12 or 3.13, Node.js 22 and Git. Create an environment:

```sh
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then:

```sh
python -m pip install -r requirements-dev.txt
npm ci
python -m playwright install chromium
python -m unittest discover -s tests -v
node --check site/app.js
node tests/test_frontend.cjs
python scripts/import_data.py --supplement-current
python scripts/import_player_data.py
python scripts/train_predict.py
python scripts/validation.py
python scripts/audit_data.py
python scripts/compare_uncertainty.py
python scripts/browser_check.py
```

On Linux CI, use `python -m playwright install --with-deps chromium`. `requirements.txt` alone installs the model dependencies; Playwright and axe-core are development checks. NumPy is constrained below 2.4 to avoid the timedelta deprecations observed with pandas 2.x and NumPy 2.5.

The first match import clones `openfootball/england` into `.cache/england`. Existing checkouts are deliberately not changed. To fetch newer match history on later runs:

```sh
git -C .cache/england pull --ff-only
python scripts/import_data.py --supplement-current
```

The supplements download the current and preceding two seasons of regular-season results from football-data.co.uk, its forthcoming League One fixtures, and the official FPL Premier League schedule. FPL provides confirmed fixture times and event numbers. Importers retain source hashes and retrieval times. Current-source failures stop the build instead of silently relabelling old data as fresh. Missing historical player seasons are recorded in the player report.

To replay downloaded bytes while this snapshot is still current:

```sh
python scripts/import_data.py --supplement-current --offline
python scripts/import_player_data.py --offline
python scripts/train_predict.py
```

This requires the same `.cache/england` revision and `.cache/downloads` files. Offline replay preserves player retrieval time: a snapshot older than 48 hours will not produce player projections. Build date still advances, so later-date outputs are intentionally different. Source hashes and input CSV hashes are in `site/data/data_quality.json` and `site/data/player_report.json`.

For a historical match-only import, use `python scripts/import_data.py --source path/to/england --as-of YYYY-MM-DD`; current supplements are forbidden with a historical as-of date. This is **not** a historical player snapshot. Do not use today's player statistics or full-season aggregates for historical player evaluation.

## Model and evaluation

The team model uses exponentially weighted home/away scoring and conceding rates, a two-year half-life, eight-year lookback, venue-specific league priors and geometric attack/defence blending. Six equivalent prior games are used normally and eighteen for cross-division fallback. The latter is a conservative heuristic, not a fitted promotion coefficient.

Calibration selects an exponent from 0.8 to 2.0 using an evenly spaced sample of at most 500 fixtures from the season preceding the final test. The last fully completed season is the final test, with every eligible fixture evaluated. Predictions use only results dated strictly before each fixture. Earlier test results may inform later test predictions (rolling origin); model parameters and calibration remain fixed. No parameter was tuned on the final test outcomes. `45ad019` supplies the original implementation for comparison on identical repaired inputs and cutoffs. Without that Git object, the report explicitly marks the direct comparison unavailable.

The baseline is per-league historical home/draw/away frequencies before test start, with add-one smoothing, held fixed throughout the test. Reports contain date ranges, sample sizes, multiclass Brier score, log loss, accuracy, top-choice confidence bins and expected calibration error, overall and per league. Player estimates are not validated by this match evaluation.

The adaptive Poisson score grid preserves probability mass. Season simulations use the same calibrated outcome probabilities as match cards; finishing ties share probability after points, goal difference and goals scored. Displayed expected goals are underlying Poisson rates, not necessarily the means after outcome calibration. Forecasts use a fixed current information date, and season simulations require a complete regular-season home/away schedule with confirmed past results. Points deductions, head-to-head rules, playoffs and future changes in team strength are not modelled.

Player history joins on persistent FPL codes, not names or season-specific IDs. Missing historical xG/xA has separate exposure denominators. Players without history use positional priors. Unavailable statuses are excluded, starting XIs require a legal formation with one goalkeeper, aggregate starting chances are capped at one goalkeeper plus ten outfield players, and scoring/assisting probabilities cannot exceed availability. These remain uncalibrated heuristics; current-season totals do not fully describe recent tactical roles or substitutions.

## Deployment readiness

`.github/workflows/deploy.yml` installs dependencies, runs Python/JavaScript tests, imports real data, rebuilds and validates the artifact, runs Chromium and accessibility checks, and uploads verification evidence. Pull requests build and test but cannot deploy. Deployment remains limited to `main` after a successful build. Reading or running the local checks never publishes the site.

The workflow passed actionlint and its build commands were run locally. A hosted GitHub Actions run and public Pages deployment were **not** performed during this validation.

## Main files

- `scripts/import_data.py`, `current_matches.py`, `source_data.py`: match parsing, supplemental sources and cached downloads.
- `scripts/import_player_data.py`: historical and current FPL data with stable identity mapping.
- `scripts/train_predict.py`, `evaluation.py`: predictions, simulations and isolated evaluation.
- `scripts/validation.py`, `audit_data.py`, `browser_check.py`: artifact, real-data and browser checks.
- `tests/`: offline regression tests.
- `site/data/`: refreshed predictions, per-fixture evaluation, source coverage and player matching reports.
- `artifacts/`: source audit, browser evidence and descriptive comparison uncertainty.

Sources: [openfootball](https://github.com/openfootball/england), [football-data.co.uk](https://www.football-data.co.uk/englandm.php), [official FPL](https://fantasy.premierleague.com/), and [historical FPL archive](https://github.com/vaastav/Fantasy-Premier-League). Independent educational analysis, not betting advice.
