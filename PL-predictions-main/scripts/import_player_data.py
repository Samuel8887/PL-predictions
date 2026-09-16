"""Build a compact, reproducible Premier League player-stat dataset.

The historical gameweek CSVs are from the community-maintained
vaastav/Fantasy-Premier-League archive.  The current-player snapshot is fetched
from Fantasy Premier League's public API.  The site deliberately stores only
the fields required for the model (rather than re-publishing the raw feeds).
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
import urllib.request
from io import BytesIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
FPL_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
SEASONS = [f"{year}-{str(year + 1)[-2:]}" for year in range(2016, date.today().year - (date.today().month < 7))]


def player_key(value: str) -> str:
    """A conservative key used only to supplement a player's own recent rates."""
    value = re.sub(r"_\d+$", "", str(value))
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def get_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "PL-predictions educational model"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def number(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0) if column in frame else pd.Series(0, index=frame.index)


def historical_rows() -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        try:
            content = get_bytes(HISTORY_URL.format(season=season))
            try:
                raw = pd.read_csv(BytesIO(content))
            except UnicodeDecodeError:
                # The earliest exports contain a small number of Latin-1 names.
                raw = pd.read_csv(BytesIO(content), encoding="latin-1")
        except Exception as error:
            print(f"Skipped {season}: {error}")
            continue
        if not {"name", "minutes", "goals_scored", "assists"}.issubset(raw.columns):
            print(f"Skipped {season}: unexpected columns")
            continue
        cleaned = pd.DataFrame({
            "player": raw["name"].astype(str),
            "player_key": raw["name"].map(player_key),
            "season": season,
            "minutes": number(raw, "minutes"),
            "starts": number(raw, "starts"),
            "goals": number(raw, "goals_scored"),
            "assists": number(raw, "assists"),
            "xg": number(raw, "expected_goals"),
            "xa": number(raw, "expected_assists"),
        })
        rows.append(cleaned.groupby(["player", "player_key", "season"], as_index=False).sum(numeric_only=True))
        print(f"Downloaded {season}: {len(raw):,} player-fixture rows")
    if not rows:
        raise RuntimeError("No historical player CSVs could be downloaded.")
    return pd.concat(rows, ignore_index=True)


def current_rows() -> pd.DataFrame:
    payload = json.loads(get_bytes(FPL_URL))
    teams = {team["id"]: team["name"] for team in payload["teams"]}
    players = pd.DataFrame(payload["elements"])
    full_name = (players.first_name.fillna("") + " " + players.second_name.fillna("")).str.strip()
    return pd.DataFrame({
        "player": full_name,
        "player_key": full_name.map(player_key),
        "team": players.team.map(teams),
        "position": players.element_type.map({1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}),
        "status": players.status,
        "chance_of_playing": pd.to_numeric(players.chance_of_playing_next_round, errors="coerce").fillna(100),
        "minutes": number(players, "minutes"),
        "starts": number(players, "starts"),
        "goals": number(players, "goals_scored"),
        "assists": number(players, "assists"),
        "xg": number(players, "expected_goals"),
        "xa": number(players, "expected_assists"),
    })


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    history = historical_rows()
    current = current_rows()
    history.to_csv(DATA / "player_history.csv", index=False)
    current.to_csv(DATA / "current_players.csv", index=False)
    print(f"Wrote {len(history):,} player-season rows and {len(current):,} current-player rows")


if __name__ == "__main__":
    main()
