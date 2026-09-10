"""Create leakage-safe Poisson predictions and a chronological evaluation report.

This intentionally compact baseline estimates a team's attack/defence from its
previous league matches with exponential time decay and Bayesian shrinkage. Each
league has its own scoring average and home advantage; there is no random split.
"""
from __future__ import annotations
import json, math
from datetime import datetime
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

def probability(lh, la):
    goals = np.arange(0, 11)
    matrix = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
    return float(np.tril(matrix, -1).sum()), float(np.trace(matrix)), float(np.triu(matrix, 1).sum())

def player_priors(history: pd.DataFrame) -> pd.DataFrame:
    """Return recency-weighted xG/xA rates for players seen in historical FPL data."""
    if history.empty: return pd.DataFrame(columns=["player_key", "prior_minutes", "prior_xg", "prior_xa", "prior_goals", "prior_assists"])
    history = history.copy()
    history["season_year"] = history.season.str[:4].astype(int)
    # 2025-26 is the latest completed season in the imported archive.
    history["weight"] = 0.55 ** (2025 - history.season_year)
    for column in ["minutes", "xg", "xa", "goals", "assists"]: history[column] *= history.weight
    return history.groupby("player_key", as_index=False).agg(
        prior_minutes=("minutes", "sum"), prior_xg=("xg", "sum"), prior_xa=("xa", "sum"),
        prior_goals=("goals", "sum"), prior_assists=("assists", "sum"))

def player_predictions(current: pd.DataFrame, priors: pd.DataFrame, fixture: pd.Series, home_goals: float, away_goals: float):
    """Allocate team xG to likely starters; it never changes the team forecast itself."""
    if current.empty: return None
    side_rows = []
    for team, expected_goals in ((fixture.home_team, home_goals), (fixture.away_team, away_goals)):
        players = current[current.model_team.eq(team) & current.position.ne("GKP") & current.minutes.gt(0)].copy()
        if players.empty: continue
        players = players.merge(priors, how="left", on="player_key")
        prior_columns = ["prior_minutes", "prior_xg", "prior_xa", "prior_goals", "prior_assists"]
        players[prior_columns] = players[prior_columns].fillna(0)
        team_starts = max(1, float(players.starts.max()))
        team_games = max(1, float(current.starts.max()))
        start_fraction = players.starts / team_starts
        minutes_fraction = (players.minutes / (90 * team_games)).clip(upper=1)
        players["start_probability"] = (0.10 + 0.90 * (0.65 * start_fraction + 0.35 * minutes_fraction)).clip(.05, .98)
        players.loc[players.status.ne("a"), "start_probability"] *= .35
        players["start_probability"] *= (players.chance_of_playing / 100).clip(0, 1)
        # A 240-minute historical prior prevents three early-season matches from
        # completely dominating, while keeping current xG/xA the strongest signal.
        prior_weight = 240.0
        goal_default = players.position.map({"DEF": .06, "MID": .16, "FWD": .28}).fillna(.12)
        assist_default = players.position.map({"DEF": .06, "MID": .14, "FWD": .09}).fillna(.10)
        players["goal_rate"] = (players.xg + .25 * players.goals + prior_weight * ((players.prior_xg + .25 * players.prior_goals) / players.prior_minutes.replace(0, np.nan)).fillna(goal_default) / 90) / (players.minutes + prior_weight)
        players["assist_rate"] = (players.xa + .20 * players.assists + prior_weight * ((players.prior_xa + .20 * players.prior_assists) / players.prior_minutes.replace(0, np.nan)).fillna(assist_default) / 90) / (players.minutes + prior_weight)
        players["goal_weight"] = (players.goal_rate * players.start_probability).clip(lower=.0001)
        players["assist_weight"] = (players.assist_rate * players.start_probability).clip(lower=.0001)
        players["goal_lambda"] = expected_goals * players.goal_weight / players.goal_weight.sum()
        # Not every goal receives an assist, so the total assist opportunity is lower.
        players["assist_lambda"] = expected_goals * .78 * players.assist_weight / players.assist_weight.sum()
        players["score_probability"] = 1 - np.exp(-players.goal_lambda)
        players["assist_probability"] = 1 - np.exp(-players.assist_lambda)
        players["performance_score"] = players.goal_lambda + players.assist_lambda
        columns = ["player", "position", "start_probability", "score_probability", "assist_probability", "performance_score"]
        stats = players[columns].round({"start_probability": 4, "score_probability": 4, "assist_probability": 4, "performance_score": 4})
        side_rows.append({
            "team": team,
            "top_performers": stats.sort_values(["performance_score", "score_probability"], ascending=False).head(3).to_dict(orient="records"),
            "top_scorer": stats.sort_values("score_probability", ascending=False).iloc[0].to_dict(),
            "top_assister": stats.sort_values("assist_probability", ascending=False).iloc[0].to_dict(),
        })
    return side_rows or None

def state_prediction(history: pd.DataFrame, fixture: pd.Series):
    """Use strictly dates before fixture date. Same-day results are excluded too."""
    prior = history[(history.league == fixture.league) & (history.date < fixture.date) & (history.date >= fixture.date - pd.Timedelta(days=LOOKBACK_DAYS))].copy()
    if len(prior) < 40: return None
    age = (fixture.date - prior.date).dt.days.clip(lower=0)
    prior["w"] = np.exp(-math.log(2) * age / HALF_LIFE_DAYS)
    # Per-league baseline avoids considering the three divisions interchangeable.
    home_base = np.average(prior.home_goals, weights=prior.w); away_base = np.average(prior.away_goals, weights=prior.w)
    league_goal = (home_base + away_base) / 2
    def team_rates(team, home):
        side = prior[prior.home_team_id.eq(team)] if home else prior[prior.away_team_id.eq(team)]
        context = "same-division"
        # Newly promoted/relegated clubs can have little recent data in this
        # division. Use their other tracked English-league matches, adjusted to
        # the target division's goal level, rather than silently dropping them.
        if len(side) < MIN_TEAM_GAMES:
            side = history[(history.date < fixture.date) & (history.date >= fixture.date - pd.Timedelta(days=LOOKBACK_DAYS))]
            side = side[side.home_team_id.eq(team)] if home else side[side.away_team_id.eq(team)]
            context = "cross-division"
        if len(side) < MIN_TEAM_GAMES: return None
        side = side.copy()
        side["w"] = np.exp(-math.log(2) * (fixture.date - side.date).dt.days.clip(lower=0) / HALF_LIFE_DAYS)
        scored = (side.home_goals if home else side.away_goals).astype(float)
        conceded = (side.away_goals if home else side.home_goals).astype(float)
        if context == "cross-division":
            source_goal = prior.groupby("league").apply(lambda x: np.average((x.home_goals + x.away_goals) / 2, weights=x.w), include_groups=False)
            factors = side.league.map(source_goal).fillna(league_goal).rdiv(league_goal)
            scored, conceded = scored * factors, conceded * factors
        # 6 equivalent prior matches makes newly promoted teams conservative.
        weight = side.w.sum(); shrink = 6
        return ((np.average(scored, weights=side.w) * weight + league_goal * shrink) / (weight + shrink),
                (np.average(conceded, weights=side.w) * weight + league_goal * shrink) / (weight + shrink), len(side), context)
    h, a = team_rates(fixture.home_team_id, True), team_rates(fixture.away_team_id, False)
    if not h or not a: return None
    # Geometric blend: a home attack meets an away defence, and vice versa.
    lh = math.sqrt((h[0] / home_base) * (a[1] / home_base)) * home_base
    la = math.sqrt((a[0] / away_base) * (h[1] / away_base)) * away_base
    hw, dr, aw = probability(lh, la)
    return {"home_win": hw, "draw": dr, "away_win": aw, "expected_home_goals": lh, "expected_away_goals": la, "history_games": min(h[2], a[2]), "context": "cross-division" if "cross-division" in (h[3], a[3]) else "same-division"}

def season_outlook(completed, future, league):
    """Simulate the remaining season and return a full position-probability table."""
    season = future.season.mode().iat[0]
    played = completed[(completed.league == league) & (completed.season == season)]
    teams = sorted(set(future.home_team) | set(future.away_team) | set(played.home_team) | set(played.away_team))
    index = {team: i for i, team in enumerate(teams)}; n = 10_000
    points = np.zeros((n, len(teams)), dtype=int); gf = np.zeros_like(points); ga = np.zeros_like(points)
    for _, m in played.iterrows():
        h, a = index[m.home_team], index[m.away_team]; hg, ag = int(m.home_goals), int(m.away_goals)
        gf[:,h] += hg; ga[:,h] += ag; gf[:,a] += ag; ga[:,a] += hg
        points[:,h] += 3 if hg > ag else 1 if hg == ag else 0; points[:,a] += 3 if ag > hg else 1 if hg == ag else 0
    rng = np.random.default_rng(20260909)
    for _, m in future.iterrows():
        pred = state_prediction(completed, m)
        if not pred: return None
        h, a = index[m.home_team], index[m.away_team]; hg, ag = rng.poisson(pred["expected_home_goals"], n), rng.poisson(pred["expected_away_goals"], n)
        gf[:,h] += hg; ga[:,h] += ag; gf[:,a] += ag; ga[:,a] += hg
        points[:,h] += (hg > ag) * 3 + (hg == ag); points[:,a] += (ag > hg) * 3 + (hg == ag)
    # Positions use the standard league ordering: points, goal difference, then goals scored.
    # An exact tie after those criteria is shared evenly so team-name order cannot affect a probability.
    position_counts = np.zeros((len(teams), len(teams)), dtype=int)
    for run in range(n):
        order = np.lexsort((-gf[run], -(gf[run] - ga[run]), -points[run]))
        start = 0
        while start < len(teams):
            end = start + 1
            team = order[start]
            while end < len(teams) and (
                points[run, order[end]] == points[run, team]
                and gf[run, order[end]] - ga[run, order[end]] == gf[run, team] - ga[run, team]
                and gf[run, order[end]] == gf[run, team]
            ):
                end += 1
            # Tied teams occupy the same range of positions; give each an equal share.
            for position in range(start, end):
                position_counts[order[start:end], position] += 1
            start = end
    table = []
    for i, team in enumerate(teams):
        probabilities = position_counts[i] / n
        most_likely_position = int(probabilities.argmax()) + 1
        table.append({
            "team": team,
            "expected_position": round(float(np.dot(probabilities, np.arange(1, len(teams) + 1))), 2),
            "most_likely_position": most_likely_position,
            "most_likely_position_probability": round(float(probabilities.max()), 4),
            "win_probability": round(float(probabilities[0]), 4),
            "position_probabilities": [round(float(value), 4) for value in probabilities],
        })
    table.sort(key=lambda row: (row["expected_position"], -row["win_probability"]))
    return {"season": season, "simulations": n, "table": table}

def outcome(hg, ag): return 0 if hg > ag else 1 if hg == ag else 2
def evaluate(completed):
    # Latest 20% is held out chronologically; predictions see only earlier dates.
    cutoff = completed.date.quantile(.80)
    test = completed[completed.date >= cutoff]
    # A deterministic evenly-spaced sample keeps daily static builds quick while
    # still testing only genuinely later fixtures from the chronological holdout.
    test = test.iloc[::max(1, math.ceil(len(test) / 600))]
    rows = []
    for _, fixture in test.iterrows():
        pred = state_prediction(completed, fixture)
        if pred: rows.append((outcome(fixture.home_goals, fixture.away_goals), [pred["home_win"], pred["draw"], pred["away_win"]]))
    if not rows: return {"message": "Not enough chronological validation data."}
    y, probs = zip(*rows); probs = np.asarray(probs); y = np.asarray(y); onehot = np.eye(3)[y]
    baseline = completed[completed.date < cutoff]
    base_p = np.array([(baseline.home_goals > baseline.away_goals).mean(), (baseline.home_goals == baseline.away_goals).mean(), (baseline.home_goals < baseline.away_goals).mean()])
    bins = []
    for lo in np.arange(0, 1, .2):
        mask = (probs.max(axis=1) >= lo) & (probs.max(axis=1) < lo + .2)
        if mask.any(): bins.append({"range": f"{lo:.1f}-{lo+.2:.1f}", "n": int(mask.sum()), "mean_confidence": round(float(probs.max(1)[mask].mean()), 3), "accuracy": round(float((probs.argmax(1)[mask] == y[mask]).mean()), 3)})
    return {"validation": "chronological final 20% (evenly-spaced sample, max 600 fixtures)", "fixtures": len(y), "accuracy": round(float((probs.argmax(1) == y).mean()), 3), "log_loss": round(float(-np.log(probs[np.arange(len(y)), y].clip(1e-12)).mean()), 3), "brier_score": round(float(((probs-onehot)**2).sum(1).mean()), 3), "baseline_accuracy": round(float((np.argmax(base_p) == y).mean()), 3), "baseline_log_loss": round(float(-np.log(base_p[y]).mean()), 3), "calibration": bins}

def main():
    matches = pd.read_csv(ROOT / "data" / "matches.csv", parse_dates=["date"])
    completed = matches[matches.status.eq("completed")].dropna(subset=["home_goals", "away_goals"]).copy()
    completed[["home_goals", "away_goals"]] = completed[["home_goals", "away_goals"]].astype(int)
    current_path, history_path = ROOT / "data" / "current_players.csv", ROOT / "data" / "player_history.csv"
    current = pd.read_csv(current_path) if current_path.exists() else pd.DataFrame()
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    if not current.empty:
        current["model_team"] = current.team.replace(FPL_TEAM_NAMES)
    priors = player_priors(history) if not history.empty else pd.DataFrame()
    report = evaluate(completed); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    predictions = {"generated_at": datetime.now().astimezone().isoformat(timespec="minutes"), "source": "openfootball/england", "leagues": {}, "season_outlook": {}}
    for league in sorted(matches.league.unique()):
        fixtures = matches[(matches.league == league) & (matches.status == "upcoming")].sort_values("date")
        cards = []
        for _, fixture in fixtures.iterrows():
            pred = state_prediction(completed, fixture)
            card = {"date": fixture.date.date().isoformat(), "matchweek": int(fixture.matchweek) if pd.notna(fixture.matchweek) else None, "kickoff": fixture.kickoff if pd.notna(fixture.kickoff) else None, "home_team": fixture.home_team, "away_team": fixture.away_team, "prediction_available": bool(pred)}
            if pred:
                card.update({k: round(v, 4) if isinstance(v, float) else v for k,v in pred.items()})
                if league == "premier-league":
                    picks = player_predictions(current, priors, fixture, pred["expected_home_goals"], pred["expected_away_goals"])
                    if picks: card["player_predictions"] = picks
            cards.append(card)
        predictions["leagues"][league] = cards
        if league == "premier-league" and len(fixtures): predictions["season_outlook"][league] = season_outlook(completed, fixtures, league)
    (OUT / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf8")
    # Keep display-only colour configuration beside the generated JSON so the
    # static frontend never needs a backend or a hard-coded second colour list.
    (OUT / "team_colours.json").write_text((ROOT / "config" / "team_colours.json").read_text(encoding="utf8"), encoding="utf8")
    print(f"Wrote {sum(map(len, predictions['leagues'].values()))} upcoming fixtures")
if __name__ == "__main__": main()
