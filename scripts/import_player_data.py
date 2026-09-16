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
import argparse
import source_data
from source_data import download, provenance

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
    return download(url)


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
        # FPL element IDs reset each season. The persistent code joins across years
        # and avoids accents, abbreviated names and unrelated namesakes.
        identities = pd.read_csv(BytesIO(get_bytes(f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/players_raw.csv')))
        codes = identities.set_index('id')['code']
        element = pd.to_numeric(raw['element'],errors='coerce') if 'element' in raw else pd.to_numeric(raw.name.str.extract(r'_(\d+)$')[0],errors='coerce')
        stable = element.map(codes)
        keys = pd.Series([f'code:{int(c)}' if pd.notna(c) else f'unmatched:{season}:{n}' for c,n in zip(stable,raw.name)],index=raw.index)
        cleaned = pd.DataFrame({
            "player": raw["name"].astype(str),
            "player_key": keys,
            "season": season,
            "minutes": number(raw, "minutes"),
            "starts": number(raw, "starts"),
            "goals": number(raw, "goals_scored"),
            "assists": number(raw, "assists"),
            "xg": number(raw, "expected_goals"),
            "xa": number(raw, "expected_assists"),
            "xg_minutes": number(raw,'minutes').where(pd.to_numeric(raw.get('expected_goals',pd.Series(index=raw.index,dtype=float)),errors='coerce').notna(),0),
            "xa_minutes": number(raw,'minutes').where(pd.to_numeric(raw.get('expected_assists',pd.Series(index=raw.index,dtype=float)),errors='coerce').notna(),0),
        })
        rows.append(cleaned.groupby(["player_key", "season"], as_index=False).agg({'player':'first',**{c:'sum' for c in ['minutes','starts','goals','assists','xg','xa','xg_minutes','xa_minutes']}}))
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
        "player_key": players.code.map(lambda c:f'code:{int(c)}'),
        "player_id": players.id,
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
    parser=argparse.ArgumentParser(); parser.add_argument('--offline',action='store_true'); args=parser.parse_args()
    source_data.OFFLINE=args.offline
    DATA.mkdir(parents=True, exist_ok=True)
    history = historical_rows()
    current = current_rows()
    if current.player_key.duplicated().any(): raise ValueError('Duplicate persistent player codes')
    history.to_csv(DATA / "player_history.csv", index=False)
    current.to_csv(DATA / "current_players.csv", index=False)
    from datetime import datetime, timezone
    from current_matches import season_for
    metadata={'fetched_at':provenance(FPL_URL)['fetched_at'], 'season':season_for(date.today()),
        'history_seasons':sorted(history.season.unique().tolist()),'history_players':int(history.player_key.nunique()),
        'missing_history_seasons':sorted(set(SEASONS)-set(history.season)),
        'current_source':{'url':FPL_URL,**provenance(FPL_URL)},
        'historical_sources':[{'url':url,**provenance(url)} for season in sorted(history.season.unique())
            for url in [HISTORY_URL.format(season=season),f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/players_raw.csv']],
        'current_players':len(current),'matched_current_players':int(current.player_key.isin(history.player_key).sum()),
        'unmatched_current_players':current.loc[~current.player_key.isin(history.player_key),['player','team']].to_dict('records'),
        'unmapped_history_rows':int(history.player_key.str.startswith('unmatched:').sum()),
        'identity':'Persistent FPL code; unknown archive identities are never name-merged.',
        'historical_player_backtest':False}
    (DATA/'player_sources.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print(f"Wrote {len(history):,} player-season rows and {len(current):,} current-player rows")


if __name__ == "__main__":
    main()
