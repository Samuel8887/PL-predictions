"""Create leakage-safe Poisson predictions and a chronological evaluation report.

This intentionally compact baseline estimates a team's attack/defence from its
previous league matches with exponential time decay and Bayesian shrinkage. Each
league has its own scoring average and home advantage; there is no random split.
"""
from __future__ import annotations
import json, math, argparse, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data"
MIN_TEAM_GAMES = 5
HALF_LIFE_DAYS = 365 * 2
LOOKBACK_DAYS = 365 * 8  # older matches have negligible weight and add no useful precision

def score_distribution(lh, la, calibration=1.0):
    """Adaptive score grid; calibration preserves score ratios within each outcome."""
    if not all(np.isfinite(x) and x >= 0 for x in (lh, la)):
        raise ValueError("Expected goals must be finite and nonnegative")
    limit = max(10, int(poisson.ppf(1 - 1e-12, max(lh, la))))
    goals = np.arange(limit + 1)
    matrix = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
    matrix /= matrix.sum()
    masks = (np.tril(np.ones_like(matrix, dtype=bool), -1),
             np.eye(len(goals), dtype=bool), np.triu(np.ones_like(matrix, dtype=bool), 1))
    raw = np.array([matrix[mask].sum() for mask in masks])
    calibrated = sharpen(raw, calibration)
    for mask, old, new in zip(masks, raw, calibrated):
        if old > 0:
            matrix[mask] *= new / old
    return matrix / matrix.sum()


def probability(lh, la):
    matrix = score_distribution(lh, la)
    return float(np.tril(matrix, -1).sum()), float(np.trace(matrix)), float(np.triu(matrix, 1).sum())

def sharpen(probabilities, exponent):
    """Calibrate outcome classes; underlying rates stay fixed, score means can change."""
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError("Calibration exponent must be positive and finite")
    values = np.asarray(probabilities, dtype=float) ** exponent
    return values / values.sum()


def player_priors(history: pd.DataFrame) -> pd.DataFrame:
    """Return recency-weighted xG/xA rates for players seen in historical FPL data."""
    if history.empty: return pd.DataFrame(columns=["player_key", "prior_minutes", "prior_xg", "prior_xa", "prior_goals", "prior_assists"])
    history = history.copy()
    history["season_year"] = history.season.str[:4].astype(int)
    history["weight"] = 0.55 ** (history.season_year.max() - history.season_year)
    for field in ['xg_minutes','xa_minutes']:
        if field not in history: history[field] = history.minutes
    for column in ["minutes", "xg", "xa", "goals", "assists",'xg_minutes','xa_minutes']: history[column] *= history.weight
    return history.groupby("player_key", as_index=False).agg(
        prior_minutes=("minutes", "sum"), prior_xg=("xg", "sum"), prior_xa=("xa", "sum"),
        prior_goals=("goals", "sum"), prior_assists=("assists", "sum"),
        prior_xg_minutes=('xg_minutes','sum'), prior_xa_minutes=('xa_minutes','sum'))

def projected_squad(players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the strongest eligible XI across plausible formations."""
    ranked = players.sort_values(["start_probability", "minutes"], ascending=False)
    candidates = []
    for defenders, midfielders, forwards in ((4,4,2), (4,3,3), (4,5,1), (3,5,2), (3,4,3), (5,3,2), (5,4,1)):
        groups = [ranked[ranked.position.eq(pos)].head(count)
                  for pos, count in (("GKP",1), ("DEF",defenders), ("MID",midfielders), ("FWD",forwards))]
        lineup = pd.concat(groups)
        if len(lineup) == 11:
            candidates.append(lineup)
    if not candidates:
        return ranked.iloc[:0].copy(), ranked.head(9).copy()
    lineup = max(candidates, key=lambda frame: frame.start_probability.sum())
    bench = ranked.drop(lineup.index).head(9)
    return lineup.copy(), bench.copy()

def player_predictions(current: pd.DataFrame, priors: pd.DataFrame, fixture: pd.Series, home_goals: float, away_goals: float):
    """Allocate team xG to likely starters; it never changes the team forecast itself."""
    if current.empty: return None
    side_rows = []
    for team, expected_goals in ((fixture.home_team, home_goals), (fixture.away_team, away_goals)):
        pool = current[current.model_team.eq(team)].copy()
        # Include unused but available squad members so the projected bench is
        # a real nine-player bench rather than a list limited to past minutes.
        players = pool[pool.status.isin(["a", "d"]) & pool.chance_of_playing.gt(0)].copy()
        if players.empty: continue
        players = players.merge(priors if not priors.empty else player_priors(pd.DataFrame()), how="left", on="player_key")
        prior_columns = ["prior_minutes", "prior_xg", "prior_xa", "prior_goals", "prior_assists"]
        players[prior_columns] = players[prior_columns].astype(float).fillna(0)
        for field in ['prior_xg_minutes','prior_xa_minutes']:
            if field not in players: players[field] = players.prior_minutes
            players[field] = players[field].fillna(0)
        team_starts = max(1, float(players.starts.max()))
        team_games = max(1, float(pool.starts.max()), float(pool.minutes.max()) / 90)
        start_fraction = players.starts / team_starts
        minutes_fraction = (players.minutes / (90 * team_games)).clip(upper=1)
        players["start_probability"] = (0.10 + 0.90 * (0.65 * start_fraction + 0.35 * minutes_fraction)).clip(.05, .98)
        players.loc[players.status.ne("a"), "start_probability"] *= .35
        players["start_probability"] *= (players.chance_of_playing / 100).clip(0, 1)
        for mask, capacity in [(players.position.eq('GKP'),1),(players.position.ne('GKP'),10)]:
            total = players.loc[mask,'start_probability'].sum()
            if total > capacity: players.loc[mask,'start_probability'] *= capacity / total
        lineup, bench = projected_squad(players)
        attackers = players[players.position.ne("GKP")].copy()
        # A 240-minute historical prior prevents three early-season matches from
        # completely dominating, while keeping current xG/xA the strongest signal.
        if attackers.empty: continue
        prior_weight = 240.0
        goal_default = attackers.position.map({"DEF": .06, "MID": .16, "FWD": .28}).fillna(.12)
        assist_default = attackers.position.map({"DEF": .06, "MID": .14, "FWD": .09}).fillna(.10)
        prior_g = attackers.prior_goals / attackers.prior_minutes.replace(0,np.nan)
        prior_a = attackers.prior_assists / attackers.prior_minutes.replace(0,np.nan)
        prior_xg = (attackers.prior_xg / attackers.prior_xg_minutes.replace(0,np.nan)).fillna(prior_g)
        prior_xa = (attackers.prior_xa / attackers.prior_xa_minutes.replace(0,np.nan)).fillna(prior_a)
        attackers["goal_rate"] = (attackers.xg + .25 * attackers.goals + prior_weight * (prior_xg + .25*prior_g).fillna(goal_default/90)) / (attackers.minutes+prior_weight)
        attackers["assist_rate"] = (attackers.xa + .20 * attackers.assists + prior_weight * (prior_xa + .20*prior_a).fillna(assist_default/90)) / (attackers.minutes+prior_weight)
        # Starter and substitute exposure, capped at 90 minutes; heuristic, not calibrated.
        attackers["expected_minutes"] = 75 * attackers.start_probability + 15 * (1 - attackers.start_probability) * (attackers.chance_of_playing / 100)
        attackers["goal_weight"] = attackers.goal_rate * attackers.expected_minutes
        attackers["assist_weight"] = attackers.assist_rate * attackers.expected_minutes
        for field in ['goal_weight','assist_weight']:
            if attackers[field].sum() <= 0: attackers[field] = attackers.expected_minutes
        attackers["goal_lambda"] = expected_goals * attackers.goal_weight / attackers.goal_weight.sum()
        # Not every goal receives an assist, so the total assist opportunity is lower.
        attackers["assist_lambda"] = expected_goals * .78 * attackers.assist_weight / attackers.assist_weight.sum()
        availability = (attackers.chance_of_playing / 100).clip(0,1)
        attackers["score_probability"] = availability * (1 - np.exp(-attackers.goal_lambda/availability))
        attackers["assist_probability"] = availability * (1 - np.exp(-attackers.assist_lambda/availability))
        attackers["performance_score"] = attackers.goal_lambda + attackers.assist_lambda
        columns = ["player", "position", "minutes", "goals", "assists", "xg", "xa", "start_probability", "score_probability", "assist_probability", "performance_score"]
        stats = attackers[columns].copy()
        metrics = ["score_probability", "assist_probability", "performance_score"]
        # Preserve row identity: joining on rounded floating-point statistics loses matches.
        lineup = lineup.join(attackers[metrics])
        bench = bench.join(attackers[metrics])
        lineup[["score_probability", "assist_probability", "performance_score"]] = lineup[["score_probability", "assist_probability", "performance_score"]].fillna(0)
        bench[["score_probability", "assist_probability", "performance_score"]] = bench[["score_probability", "assist_probability", "performance_score"]].fillna(0)
        side_rows.append({
            "team": team,
            "top_performers": stats.sort_values(["performance_score", "score_probability"], ascending=False).head(3).to_dict(orient="records"),
            "top_scorer": stats.sort_values("score_probability", ascending=False).iloc[0].to_dict(),
            "top_assister": stats.sort_values("assist_probability", ascending=False).iloc[0].to_dict(),
            "projected_lineup": lineup[columns].round({"xg": 2, "xa": 2, "start_probability": 4, "score_probability": 4, "assist_probability": 4}).to_dict(orient="records"),
            "projected_bench": bench[columns].round({"xg": 2, "xa": 2, "start_probability": 4, "score_probability": 4, "assist_probability": 4}).to_dict(orient="records"),
        })
    return side_rows or None

def state_prediction(history: pd.DataFrame, fixture: pd.Series, calibration: float = 1.0):
    """Use strictly dates before fixture date. Same-day results are excluded too."""
    if 'status' in history: history = history[history.status.eq('completed')]
    history = history.dropna(subset=['home_goals','away_goals'])
    prior = history[(history.league == fixture.league) & (history.date < fixture.date) & (history.date >= fixture.date - pd.Timedelta(days=LOOKBACK_DAYS))].copy()
    if len(prior) < 40: return None
    age = (fixture.date - prior.date).dt.days.clip(lower=0)
    prior["w"] = np.exp(-math.log(2) * age / HALF_LIFE_DAYS)
    # Per-league baseline avoids considering the three divisions interchangeable.
    home_base = np.average(prior.home_goals, weights=prior.w); away_base = np.average(prior.away_goals, weights=prior.w)
    def team_rates(team, home):
        side = prior[prior.home_team_id.eq(team)] if home else prior[prior.away_team_id.eq(team)]
        context = "same-division"
        # Newly promoted/relegated clubs can have little recent data in this
        # division. Use their other tracked English-league matches with stronger
        # target-division shrinkage; no learned promotion coefficient is available.
        if len(side) < MIN_TEAM_GAMES:
            side = history[(history.date < fixture.date) & (history.date >= fixture.date - pd.Timedelta(days=LOOKBACK_DAYS))]
            side = side[side.home_team_id.eq(team)] if home else side[side.away_team_id.eq(team)]
            context = "cross-division"
        if len(side) < MIN_TEAM_GAMES: return None
        side = side.copy()
        side["w"] = np.exp(-math.log(2) * (fixture.date - side.date).dt.days.clip(lower=0) / HALF_LIFE_DAYS)
        scored = (side.home_goals if home else side.away_goals).astype(float)
        conceded = (side.away_goals if home else side.home_goals).astype(float)
        # Division scoring means are not division strength. With no fitted transition
        # coefficient, increase shrinkage rather than inventing a promotion adjustment.
        weight = side.w.sum(); shrink = 18 if context == "cross-division" else 6
        return ((np.average(scored, weights=side.w) * weight + (home_base if home else away_base) * shrink) / (weight + shrink),
                (np.average(conceded, weights=side.w) * weight + (away_base if home else home_base) * shrink) / (weight + shrink), len(side), context)
    h, a = team_rates(fixture.home_team_id, True), team_rates(fixture.away_team_id, False)
    if not h or not a: return None
    # Geometric blend: a home attack meets an away defence, and vice versa.
    lh = math.sqrt(h[0] * a[1])
    la = math.sqrt(a[0] * h[1])
    hw, dr, aw = sharpen(probability(lh, la), calibration)
    return {"home_win": hw, "draw": dr, "away_win": aw, "expected_home_goals": lh, "expected_away_goals": la, "history_games": min(h[2], a[2]), "context": "cross-division" if "cross-division" in (h[3], a[3]) else "same-division"}

def season_outlook(completed, future, league, calibration=1.0, predictions=None, simulations=10_000):
    """Simulate the remaining season and return a full position-probability table."""
    if future.empty: return None
    future = future[future.league.eq(league)]
    if future.empty: return None
    if future.season.nunique() != 1: raise ValueError('Season outlook requires exactly one season')
    season = future.season.iat[0]
    played = completed[(completed.league == league) & (completed.season == season)]
    teams = sorted(set(future.home_team) | set(future.away_team) | set(played.home_team) | set(played.away_team))
    index = {team: i for i, team in enumerate(teams)}; n = simulations
    points = np.zeros((n, len(teams)), dtype=int); gf = np.zeros_like(points); ga = np.zeros_like(points)
    for _, m in played.iterrows():
        h, a = index[m.home_team], index[m.away_team]; hg, ag = int(m.home_goals), int(m.away_goals)
        gf[:,h] += hg; ga[:,h] += ag; gf[:,a] += ag; ga[:,a] += hg
        points[:,h] += 3 if hg > ag else 1 if hg == ag else 0; points[:,a] += 3 if ag > hg else 1 if hg == ag else 0
    rng = np.random.default_rng(20260909)
    for fixture_index, m in future.iterrows():
        forecast=m.copy()
        forecast.date=min(m.date,pd.Timestamp.now(tz='Europe/London').normalize().tz_localize(None))
        pred = predictions.get(fixture_index) if predictions is not None else state_prediction(completed, forecast, calibration)
        if not pred: return None
        h, a = index[m.home_team], index[m.away_team]
        scores = score_distribution(pred["expected_home_goals"], pred["expected_away_goals"], calibration)
        sampled = rng.choice(scores.size, size=n, p=scores.ravel())
        hg, ag = sampled // scores.shape[1], sampled % scores.shape[1]
        gf[:,h] += hg; ga[:,h] += ag; gf[:,a] += ag; ga[:,a] += hg
        points[:,h] += (hg > ag) * 3 + (hg == ag); points[:,a] += (ag > hg) * 3 + (hg == ag)
    # Positions use the standard league ordering: points, goal difference, then goals scored.
    # An exact tie after those criteria is shared evenly so team-name order cannot affect a probability.
    position_counts = np.zeros((len(teams), len(teams)), dtype=float)
    orders = np.lexsort((-gf, -(gf - ga), -points), axis=1)
    for position in range(len(teams)):
        position_counts[:, position] = np.bincount(orders[:, position], minlength=len(teams))
    ordered_points = np.take_along_axis(points, orders, axis=1)
    ordered_gd = np.take_along_axis(gf - ga, orders, axis=1)
    ordered_gf = np.take_along_axis(gf, orders, axis=1)
    ties = (np.diff(ordered_points, axis=1) == 0) & (np.diff(ordered_gd, axis=1) == 0) & (np.diff(ordered_gf, axis=1) == 0)
    # Most runs have no exact ties: handle only exceptional rows in Python.
    for run in np.flatnonzero(ties.any(axis=1)):
        start = 0
        while start < len(teams):
            end = start + 1
            while end < len(teams) and ties[run, end - 1]:
                end += 1
            if end - start > 1:
                group = orders[run, start:end]
                position_counts[group, np.arange(start, end)] -= 1
                position_counts[np.ix_(group, np.arange(start, end))] += 1 / (end - start)
            start = end
    table = []
    for i, team in enumerate(teams):
        probabilities = position_counts[i] / n
        most_likely_position = int(probabilities.argmax()) + 1
        table.append({
            "team": team,
            "expected_points": round(float(points[:, i].mean()), 1),
            "points_interval_80": [int(x) for x in np.quantile(points[:, i], [.1, .9])],
            "expected_position": round(float(np.dot(probabilities, np.arange(1, len(teams) + 1))), 2),
            "most_likely_position": most_likely_position,
            "most_likely_position_probability": round(float(probabilities.max()), 4),
            "win_probability": round(float(probabilities[0]), 4),
            "position_probabilities": [round(float(value), 4) for value in probabilities],
        })
    table.sort(key=lambda row: (row["expected_position"], -row["win_probability"]))
    return {"season": season, "simulations": n, "tie_method": "Equal shares after points, goal difference and goals scored; head-to-head and playoffs not modelled.", "table": table}

def outcome(hg, ag): return 0 if hg > ag else 1 if hg == ag else 2
def evaluate(completed):
    from evaluation import evaluate_models
    import sys
    return evaluate_models(completed,sys.modules[__name__],ROOT)[0]


def main(output=None, compare=True):
    from current_matches import season_for, canonical
    from validation import coverage, validate_predictions
    from evaluation import evaluate_models
    import sys
    out = Path(output) if output else OUT
    today = pd.Timestamp.now(tz='Europe/London').normalize().tz_localize(None)
    season = season_for(today.date())
    matches = pd.read_csv(ROOT / "data" / "matches.csv")
    matches["date"] = pd.to_datetime(matches["date"], format="mixed", errors="raise")
    completed = matches[matches.status.eq("completed") & (matches.date < today)].dropna(subset=["home_goals", "away_goals"]).copy()
    completed = completed.sort_values("date").reset_index(drop=True)
    completed[["home_goals", "away_goals"]] = completed[["home_goals", "away_goals"]].astype(int)
    current_path, history_path = ROOT / "data" / "current_players.csv", ROOT / "data" / "player_history.csv"
    current = pd.read_csv(current_path) if current_path.exists() else pd.DataFrame()
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    if not current.empty:
        current["model_team"] = current.team.map(canonical)
    if not history.empty: history = history[history.season.str[:4].astype(int) < int(season[:4])]
    priors = player_priors(history) if not history.empty else pd.DataFrame()
    report, evaluation_rows = evaluate_models(completed,sys.modules[__name__],ROOT,compare=compare)
    quality = coverage(matches,season)
    player_path = ROOT / 'data/player_sources.json'
    player_report = json.loads(player_path.read_text(encoding='utf8')) if player_path.exists() else {'message':'Player snapshot provenance unavailable.'}
    snapshot_time = pd.Timestamp(player_report.get('fetched_at','1900-01-01T00:00Z'))
    player_fresh = player_report.get('season') == season and pd.Timedelta(0) <= pd.Timestamp.now(tz='UTC')-snapshot_time <= pd.Timedelta(hours=48)
    pl = matches[matches.league.eq('premier-league') & matches.season.eq(season)]
    pl_teams=set(pl.home_team)|set(pl.away_team)
    player_report['unmatched_teams'] = sorted(set(current.model_team)-pl_teams) if not current.empty else []
    player_report['projection_enabled'] = player_fresh and not player_report['unmatched_teams']
    if not player_report['projection_enabled']: current=pd.DataFrame()

    # Use the exponent selected before the untouched final validation period,
    # rather than re-fitting on every completed result and over-sharpening odds.
    probability_sharpness = report.get("probability_sharpness", 1.0)

    predictions = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"), "model_version": "2.1", "as_of": str(today.date()), "season": season, "coverage": quality, "player_snapshot": {k:player_report.get(k) for k in ["fetched_at","season","projection_enabled"]}, "snapshot_kind": "rebuilt", "source": "openfootball/england; football-data.co.uk; official FPL", "leagues": {}, "season_outlook": {}}
    for league in ["premier-league","championship","league-one"]:
        fixtures = matches[(matches.league == league) & (matches.status == "upcoming") & matches.season.eq(season) & (matches.date >= today)].sort_values(['date','kickoff','home_team'],na_position='last')
        cards = []
        forecast_cache = {}
        for fixture_index, fixture in fixtures.iterrows():
            # Forecast all future fixtures from today's information set. Future
            # dates must not artificially age today's team ratings towards the prior.
            forecast_fixture = fixture.copy()
            forecast_fixture.date = min(fixture.date, today)
            pred = state_prediction(completed, forecast_fixture, probability_sharpness)
            forecast_cache[fixture_index] = pred
            card = {"date": fixture.date.date().isoformat(), "matchweek": int(fixture.matchweek) if pd.notna(fixture.matchweek) else None, "kickoff": fixture.kickoff if pd.notna(fixture.kickoff) else None, "home_team": fixture.home_team, "away_team": fixture.away_team, "prediction_available": bool(pred)}
            if pred:
                card.update({k: round(v, 4) if isinstance(v, float) else v for k,v in pred.items()})
                if league == "premier-league" and fixture.date <= today + pd.Timedelta(days=14):
                    picks = player_predictions(current, priors, fixture, pred["expected_home_goals"], pred["expected_away_goals"])
                    if picks: card["player_predictions"] = picks
            cards.append(card)
        predictions["leagues"][league] = cards
        if quality[league]["season_forecast_available"] and len(fixtures): predictions["season_outlook"][league] = season_outlook(completed, fixtures, league, probability_sharpness, forecast_cache)
    validate_predictions(predictions)
    out.mkdir(parents=True,exist_ok=True)
    sources_path=ROOT/'data/match_sources.json'
    artifacts={'predictions.json':predictions,'evaluation.json':report,'evaluation_fixtures.json':evaluation_rows,
        'player_report.json':player_report,'data_quality.json':{'coverage':quality,'sources':json.loads(sources_path.read_text()) if sources_path.exists() else {},
            'input_sha256':{name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest() for name in ['matches.csv','current_players.csv','player_history.csv'] if (ROOT/'data'/name).exists()}}}
    # Serialize and validate all outputs before replacing any previously published preview.
    encoded={name:json.dumps(value,separators=(',',':'),allow_nan=False) for name,value in artifacts.items()}
    for name,content in encoded.items():
        temporary=out/(name+'.tmp');temporary.write_text(content,encoding='utf8');temporary.replace(out/name)
    (out/'team_colours.json').write_text((ROOT/'config/team_colours.json').read_text(encoding='utf8'),encoding='utf8')
    print(f"Wrote {sum(map(len, predictions['leagues'].values()))} upcoming fixtures")
if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--no-compare',action='store_true');args=parser.parse_args()
    main(args.output,not args.no_compare)
