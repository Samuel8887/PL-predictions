"""Download and parse openfootball/england Football.TXT season files.

The parser deliberately keeps incomplete lines: a fixture is only completed when it
has a score *and* its date is not after the build date. This prevents accidental
future-result leakage from a source file that is edited ahead of time.
"""
from __future__ import annotations
import argparse, json, re, subprocess
from datetime import date, datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = {"1-premierleague.txt": "premier-league", "2-championship.txt": "championship", "3-league1.txt": "league-one"}
DATE_RE = re.compile(r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?:[A-Z][a-z]{2}\s+)?[A-Z][a-z]{2}\s+\d{1,2}(?:\s+\d{4})?$")
TIME_RE = re.compile(r"^\s*(\d{1,2}:\d{2})\s+(.*)$")
SCORE_RE = re.compile(r"\s+(\d+)\s*-\s*(\d+)\s*(?:\([^)]*\))?\s*$")

def season_start(season: str) -> int: return int(season[:4])
def load_aliases(): return json.loads((ROOT / "config/team_aliases.json").read_text(encoding="utf8"))
def normalise_team(name: str, aliases: dict[str, str]) -> str:
    """Remove only legal-style terminal suffixes; keep meaningful club names intact."""
    name = aliases.get(name, name)
    return re.sub(r"\s+(?:FC|AFC)$", "", name).strip()
def team_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
def parse_day(line: str, season: str) -> date | None:
    text = line.strip()
    if not DATE_RE.match(text): return None
    bits = text.split()
    # Most historical files omit the year. Jan--Jun belongs to season end year.
    year = int(bits[-1]) if bits[-1].isdigit() and len(bits[-1]) == 4 else None
    fragment = " ".join(bits[1:-1] if year else bits[1:])
    # Parse against a leap year first so a Feb 29 fixture is accepted.
    month_day = datetime.strptime(f"2000 {fragment}", "%Y %b %d")
    actual_year = year or season_start(season) + (1 if month_day.month <= 6 else 0)
    return date(actual_year, month_day.month, month_day.day)

def parse_fixture(text: str):
    kickoff = None
    m = TIME_RE.match(text)
    if m: kickoff, text = m.groups()
    # Modern format: Home v Away [score]; old format: Home score Away.
    # Old files put the score in the middle; modern files put it after the away side.
    old = re.match(r"^(.+?)\s+(\d+)\s*-\s*(\d+)\s*(?:\([^)]*\))?\s+(.+)$", text)
    if old and " v " not in text:
        home, hg, ag, away = old.groups()
        return home.strip(), away.strip(), kickoff, int(hg), int(ag)
    score = SCORE_RE.search(text)
    goals = (int(score.group(1)), int(score.group(2))) if score else (None, None)
    core = text[:score.start()].strip() if score else text.strip()
    if " v " in core:
        home, away = core.split(" v ", 1)
    else: return None
    return home.strip(), away.strip(), kickoff, *goals

def parse_repo(source: Path, build_date: date) -> pd.DataFrame:
    aliases, rows = load_aliases(), []
    for folder in sorted(source.iterdir()):
        if not re.match(r"^20\d\d-\d\d$", folder.name): continue
        for filename, league in LEAGUES.items():
            file = folder / filename
            if not file.exists(): continue
            current_date, matchweek = None, None
            for raw in file.read_text(encoding="utf8").splitlines():
                line = raw.strip()
                md = re.search(r"Matchday\s+(\d+)", line)
                if md: matchweek = int(md.group(1)); continue
                parsed_day = parse_day(line, folder.name)
                if parsed_day: current_date = parsed_day; continue
                if not current_date or not line or line.startswith(("#", "=", "▪")): continue
                fixture = parse_fixture(raw)
                if not fixture: continue
                home, away, kickoff, hg, ag = fixture
                home, away = normalise_team(home, aliases), normalise_team(away, aliases)
                completed = hg is not None and current_date <= build_date
                rows.append({"date": current_date.isoformat(), "season": folder.name, "league": league,
                  "matchweek": matchweek, "kickoff": kickoff, "home_team": home, "away_team": away,
                  "home_team_id": team_id(home), "away_team_id": team_id(away), "home_goals": hg if completed else None,
                  "away_goals": ag if completed else None, "status": "completed" if completed else ("upcoming" if current_date >= build_date else "incomplete")})
    if not rows:
        raise ValueError("No fixtures parsed; check the source directory and season format")
    return pd.DataFrame(rows).drop_duplicates(subset=["league", "season", "home_team_id", "away_team_id", "date"]).sort_values(["date", "league", "matchweek"], na_position="last")

def main():
    p = argparse.ArgumentParser(); p.add_argument("--source", type=Path, default=ROOT / ".cache" / "england"); p.add_argument("--output", type=Path, default=ROOT / "data" / "matches.csv"); p.add_argument("--as-of", default=date.today().isoformat()); args = p.parse_args()
    if not args.source.exists():
        args.source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/openfootball/england.git", str(args.source)], check=True)
    data = parse_repo(args.source, date.fromisoformat(args.as_of)); args.output.parent.mkdir(parents=True, exist_ok=True); data.to_csv(args.output, index=False)
    print(f"Parsed {len(data):,} fixtures ({(data.status == 'completed').sum():,} completed) into {args.output}")
if __name__ == "__main__": main()
