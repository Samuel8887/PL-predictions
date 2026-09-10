# English Football Estimates

A small, static, educational website that presents league-aware statistical estimates for upcoming English Premier League, Championship, and League One fixtures. It is **not betting advice**. The site intentionally shows no estimate when it lacks enough prior data.

## Source inspection and format

The project reads the public-domain [openfootball/england](https://github.com/openfootball/england) repository directly. Its top level has a directory for each season (for example `2004-05`, `2025-26`, and `2026-27`). The three files used in each season directory are:

| Competition | File |
| --- | --- |
| Premier League | `1-premierleague.txt` |
| Championship | `2-championship.txt` |
| League One | `3-league1.txt` |

They are Football.TXT files with a league heading, `Matchday N` headings, date lines, and indented match lines. Historical matches use `Home 2-1 (half-time) Away`; modern files may use `Home v Away 2-1 (half-time)`. Date and kick-off time may be omitted on a continuation line. `scripts/import_data.py` handles both forms and creates `data/matches.csv` with date, season, league, matchweek, teams, stable normalised IDs, goals, and status.

Only scored matches dated on or before the build date are marked `completed`; later scored lines are still treated as upcoming. This guard is deliberate protection against future-result leakage.

## Model

`scripts/train_predict.py` uses a transparent, league-aware Poisson baseline. For each fixture it uses only matches before that fixture's date (same-day matches are excluded), exponentially downweights older matches with a two-year half-life, and shrinks sparse team rates towards the league average. A team's home attack is blended with its opponent's away defence to get expected home goals; the mirror calculation produces expected away goals. Poisson score probabilities then yield home-win, draw, and away-win estimates.

For Premier League player estimates, `scripts/import_player_data.py` downloads historical gameweek records from the [vaastav/Fantasy-Premier-League archive](https://github.com/vaastav/Fantasy-Premier-League) (2016-17 onward) and the current public Fantasy Premier League feed. FBref currently blocks automated access from this build environment, so the project does not circumvent that protection or claim that its data was downloaded from FBref. The player forecast uses recent player minutes, goals, assists, xG/xA where present, and a recency-weighted historical prior to allocate the already-modelled team goal expectation across likely starters. It is a performance estimate, not a confirmed-lineup, injury, or betting prediction.

The report in `site/data/evaluation.json` uses the final chronological 20% of completed fixtures. It includes accuracy, log loss, Brier score, confidence-bin calibration, and a historical-outcome baseline. This is a baseline for learning, not a claim of predictive superiority.

`config/team_aliases.json` contains explicit safe aliases and `config/team_colours.json` contains optional display accents. The parser only removes a terminal legal-style `FC`/`AFC` suffix and never strips meaningful words such as `United`; aliases are intentional so distinct historical club identities are not merged by accident.

## Run locally

Requires Python 3.10+ and Git.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/import_data.py
python scripts/import_player_data.py
python scripts/train_predict.py
python -m http.server 8000 --directory site
```

Open `http://localhost:8000`. For a repeatable historical run, pass `--as-of 2025-06-01` to the importer. To reuse an existing source checkout, use `--source path/to/england`.

## GitHub Pages deployment

The scheduled `.github/workflows/deploy.yml` downloads the source repository, builds JSON, and deploys the `site` directory with the official Pages actions. In GitHub repository settings, set **Pages → Source** to **GitHub Actions** once. Push to `main`, use **Run workflow**, or wait for its daily schedule. The `deploy` job URL is the public site URL.

## Project structure

```
scripts/import_data.py       # downloader and Football.TXT parser
scripts/import_player_data.py # historical/current FPL player-stat importer
scripts/train_predict.py     # leak-safe model, evaluation and JSON output
config/                      # explicit aliases and display colours
data/matches.csv             # generated clean data (not required in git)
site/                        # dependency-free frontend and generated JSON
.github/workflows/deploy.yml # scheduled Pages build/deploy
```

## Roadmap

Keep match estimates first. Future additions can include player minutes/starting probability, goals and assists, xG/xA, shots and chances created, clean-sheet probability, fixture difficulty, injuries/suspensions, Fantasy Premier League point projections, and player form/opponent strength.
