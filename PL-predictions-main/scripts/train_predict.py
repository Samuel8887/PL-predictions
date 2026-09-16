"""Create leakage-safe Poisson predictions and a chronological evaluation report.

This intentionally compact baseline estimates a team's attack/defence from its
previous league matches with exponential time decay and Bayesian shrinkage. Each
league has its own scoring average and home advantage; there is no random split.
"""
from __future__ import annotations
import json, math
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
FPL_TEAM_NAMES = {
    "Brighton": "Brighton & Hove Albion", "Leeds": "Leeds United", "Man City": "Manchester City",
    "Man Utd": "Manchester United", "Newcastle": "Newcastle United", "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham Hotspur",
}

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
    """Calibrate the three-result distribution without altering expected goals."""
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError("Calibration exponent must be positive and finite")
    values = np.asarray(probabilities, dtype=float) ** exponent
    return values / values.sum()

def fit_probability_sharpness(completed: pd.DataFrame) -> float:
    """Pick a calibration exponent on an earlier chronological validation slice."""
    if len(completed) < 500: return 1.0
    start = completed.date.quantile(.75)
    sample = completed[completed.date >= start]
    sample = sample.iloc[::max(1, math.ceil(len(sample) / 500))]
    rows = []
    for _, fixture in sample.iterrows():
        prediction = state_prediction(completed, fixture)
        if prediction:
            rows.append((outcome(fixture.home_goals, fixture.away_goals), [prediction["home_win"], prediction["draw"], prediction["away_win"]]))
    if not rows: return 1.0
    y, raw = zip(*rows); y, raw = np.asarray(y), np.asarray(raw)
    candidates = np.arange(.8, 2.01, .1)
    losses = [float(-np.log(np.array([sharpen(p, value) for p in raw])[np.arange(len(y)), y].clip(1e-12)).mean()) for value in candidates]
    return round(float(candidates[int(np.argmin(losses))]), 1)

def player_priors(history: pd.DataFrame) -> pd.DataFrame:
    """Return recency-weighted xG/xA rates for players seen in historical FPL data."""
    if history.empty: return pd.DataFrame(columns=["player_key", "prior_minutes", "prior_xg", "prior_xa", "prior_goals", "prior_assists"])
    history = history.copy()
    history["season_year"] = history.season.str[:4].astype(int)
    history["weight"] = 0.55 ** (history.season_year.max() - history.season_year)
    for column in ["minutes", "xg", "xa", "goals", "assists"]: history[column] *= history.weight
    return history.groupby("player_key", as_index=False).agg(
        prior_minutes=("minutes", "sum"), prior_xg=("xg", "sum"), prior_xa=("xa", "sum"),
        prior_goals=("goals", "sum"), prior_assists=("assists", "sum"))

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
        team_starts = max(1, float(players.starts.max()))
        team_games = max(1, float(pool.starts.max()), float(pool.minutes.max()) / 90)
        start_fraction = players.starts / team_starts
        minutes_fraction = (players.minutes / (90 * team_games)).clip(upper=1)
        players["start_probability"] = (0.10 + 0.90 * (0.65 * start_fraction + 0.35 * minutes_fraction)).clip(.05, .98)
        players.loc[players.status.ne("a"), "start_probability"] *= .35
        players["start_probability"] *= (players.chance_of_playing / 100).clip(0, 1)
        lineup, bench = projected_squad(players)
        attackers = players[players.position.ne("GKP")].copy()
        # A 240-minute historical prior prevents three early-season matches from
        # completely dominating, while keeping current xG/xA the strongest signal.
        if attackers.empty: continue
        prior_weight = 240.0
        goal_default = attackers.position.map({"DEF": .06, "MID": .16, "FWD": .28}).fillna(.12)
        assist_default = attackers.position.map({"DEF": .06, "MID": .14, "FWD": .09}).fillna(.10)
        attackers["goal_rate"] = (attackers.xg + .25 * attackers.goals + prior_weight * ((attackers.prior_xg + .25 * attackers.prior_goals) / attackers.prior_minutes.replace(0, np.nan)).fillna(goal_default / 90)) / (attackers.minutes + prior_weight)
        attackers["assist_rate"] = (attackers.xa + .20 * attackers.assists + prior_weight * ((attackers.prior_xa + .20 * attackers.prior_assists) / attackers.prior_minutes.replace(0, np.nan)).fillna(assist_default / 90)) / (attackers.minutes + prior_weight)
        # Starter and substitute exposure, capped at 90 minutes; heuristic, not calibrated.
        attackers["expected_minutes"] = 75 * attackers.start_probability + 15 * (1 - attackers.start_probability) * (attackers.chance_of_playing / 100)
        attackers["goal_weight"] = attackers.goal_rate * attackers.expected_minutes
        attackers["assist_weight"] = attackers.assist_rate * attackers.expected_minutes
        attackers["goal_lambda"] = expected_goals * attackers.goal_weight / attackers.goal_weight.sum()
        # Not every goal receives an assist, so the total assist opportunity is lower.
        attackers["assist_lambda"] = expected_goals * .78 * attackers.assist_weight / attackers.assist_weight.sum()
        attackers["score_probability"] = 1 - np.exp(-attackers.goal_lambda)
        attackers["assist_probability"] = 1 - np.exp(-attackers.assist_lambda)
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
    season = future.season.mode().iat[0]
    future = future[future.season.eq(season) & future.league.eq(league)]
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
        pred = predictions.get(fixture_index) if predictions is not None else state_prediction(completed, m, calibration)
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
    completed = completed.sort_values("date")
    if completed.empty:
        return {"message": "Not enough chronological validation data."}
    # Latest 20% is held out chronologically; predictions see only earlier dates.
    cutoff = completed.date.quantile(.80)
    calibration = fit_probability_sharpness(completed[completed.date < cutoff])
    test = completed[completed.date >= cutoff]
    # A deterministic evenly-spaced sample keeps daily static builds quick while
    # still testing only genuinely later fixtures from the chronological holdout.
    test = test.iloc[::max(1, math.ceil(len(test) / 600))]
    rows = []
    for _, fixture in test.iterrows():
        pred = state_prediction(completed, fixture, calibration)
        if pred: rows.append((outcome(fixture.home_goals, fixture.away_goals), [pred["home_win"], pred["draw"], pred["away_win"]]))
    if not rows: return {"message": "Not enough chronological validation data."}
    y, probs = zip(*rows); probs = np.asarray(probs); y = np.asarray(y); onehot = np.eye(3)[y]
    baseline = completed[completed.date < cutoff]
    base_p = np.array([(baseline.home_goals > baseline.away_goals).mean(), (baseline.home_goals == baseline.away_goals).mean(), (baseline.home_goals < baseline.away_goals).mean()])
    bins = []
    for lo in np.arange(0, 1, .2):
        mask = (probs.max(axis=1) >= lo) & (probs.max(axis=1) < lo + .2)
        if mask.any(): bins.append({"range": f"{lo:.1f}-{lo+.2:.1f}", "n": int(mask.sum()), "mean_confidence": round(float(probs.max(1)[mask].mean()), 3), "accuracy": round(float((probs.argmax(1)[mask] == y[mask]).mean()), 3)})
    return {"validation": "chronological final 20% (evenly-spaced sample, max 600 fixtures)", "fixtures": len(y), "probability_sharpness": calibration, "accuracy": round(float((probs.argmax(1) == y).mean()), 3), "log_loss": round(float(-np.log(probs[np.arange(len(y)), y].clip(1e-12)).mean()), 3), "brier_score": round(float(((probs-onehot)**2).sum(1).mean()), 3), "baseline_accuracy": round(float((np.argmax(base_p) == y).mean()), 3), "baseline_log_loss": round(float(-np.log(base_p[y]).mean()), 3), "calibration": bins}

def main():
    matches = pd.read_csv(ROOT / "data" / "matches.csv")
    matches["date"] = pd.to_datetime(matches["date"], format="mixed", errors="raise")
    completed = matches[matches.status.eq("completed")].dropna(subset=["home_goals", "away_goals"]).copy()
    completed = completed.sort_values("date").reset_index(drop=True)
    completed[["home_goals", "away_goals"]] = completed[["home_goals", "away_goals"]].astype(int)
    current_path, history_path = ROOT / "data" / "current_players.csv", ROOT / "data" / "player_history.csv"
    current = pd.read_csv(current_path) if current_path.exists() else pd.DataFrame()
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    if not current.empty:
        current["model_team"] = current.team.replace(FPL_TEAM_NAMES)
    priors = player_priors(history) if not history.empty else pd.DataFrame()
    report = evaluate(completed)
    # Use the exponent selected before the untouched final validation period,
    # rather than re-fitting on every completed result and over-sharpening odds.
    probability_sharpness = report.get("probability_sharpness", 1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    predictions = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"), "model_version": "2.0", "snapshot_kind": "rebuilt", "source": "openfootball/england", "leagues": {}, "season_outlook": {}}
    for league in sorted(matches.league.unique()):
        fixtures = matches[(matches.league == league) & (matches.status == "upcoming")].sort_values("date")
        cards = []
        forecast_cache = {}
        for fixture_index, fixture in fixtures.iterrows():
            # Forecast all future fixtures from today's information set. Future
            # dates must not artificially age today's team ratings towards the prior.
            forecast_fixture = fixture.copy()
            forecast_fixture.date = min(fixture.date, pd.Timestamp.now().normalize())
            pred = state_prediction(completed, forecast_fixture, probability_sharpness)
            forecast_cache[fixture_index] = pred
            card = {"date": fixture.date.date().isoformat(), "matchweek": int(fixture.matchweek) if pd.notna(fixture.matchweek) else None, "kickoff": fixture.kickoff if pd.notna(fixture.kickoff) else None, "home_team": fixture.home_team, "away_team": fixture.away_team, "prediction_available": bool(pred)}
            if pred:
                card.update({k: round(v, 4) if isinstance(v, float) else v for k,v in pred.items()})
                if league == "premier-league" and fixture.date <= pd.Timestamp.now().normalize() + pd.Timedelta(days=14):
                    picks = player_predictions(current, priors, fixture, pred["expected_home_goals"], pred["expected_away_goals"])
                    if picks: card["player_predictions"] = picks
            cards.append(card)
        predictions["leagues"][league] = cards
        if league == "premier-league" and len(fixtures): predictions["season_outlook"][league] = season_outlook(completed, fixtures, league, probability_sharpness, forecast_cache)
    (OUT / "predictions.json").write_text(json.dumps(predictions, separators=(",", ":"), allow_nan=False), encoding="utf8")
    # Keep display-only colour configuration beside the generated JSON so the
    # static frontend never needs a backend or a hard-coded second colour list.
    (OUT / "team_colours.json").write_text((ROOT / "config" / "team_colours.json").read_text(encoding="utf8"), encoding="utf8")
    print(f"Wrote {sum(map(len, predictions['leagues'].values()))} upcoming fixtures")
if __name__ == "__main__": main()
