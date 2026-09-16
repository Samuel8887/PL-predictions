"""Supplement incomplete openfootball coverage without inventing fixtures."""
import json
from io import BytesIO
import pandas as pd
from source_data import download, provenance
from import_data import normalise_team, team_id, load_aliases

NAMES = {
    'Bristol Rvs':'Bristol Rovers','Exeter':'Exeter City','Northampton':'Northampton Town',
    'Rotherham':'Rotherham United','Shrewsbury':'Shrewsbury Town',
    'Birmingham':'Birmingham City', 'Blackburn':'Blackburn Rovers', 'Bolton':'Bolton Wanderers',
    'Bradford':'Bradford City', 'Brighton':'Brighton & Hove Albion', 'Burton':'Burton Albion',
    'Cambridge':'Cambridge United', 'Cardiff':'Cardiff City', 'Charlton':'Charlton Athletic',
    'Coventry':'Coventry City', 'Derby':'Derby County', 'Doncaster':'Doncaster Rovers',
    'Huddersfield':'Huddersfield Town', 'Hull':'Hull City', 'Ipswich':'Ipswich Town',
    'Leeds':'Leeds United', 'Leicester':'Leicester City', 'Lincoln':'Lincoln City',
    'Luton':'Luton Town', 'Man City':'Manchester City', 'Man United':'Manchester United',
    'Man Utd':'Manchester United', 'Mansfield':'Mansfield Town', 'Newcastle':'Newcastle United',
    'Norwich':'Norwich City', 'Nott\'m Forest':'Nottingham Forest', 'Nott\'ham Forest':'Nottingham Forest',
    'Oxford':'Oxford United', 'Peterboro':'Peterborough United', 'Plymouth':'Plymouth Argyle',
    'Preston':'Preston North End', 'QPR':'Queens Park Rangers', 'Sheffield Weds':'Sheffield Wednesday',
    'Spurs':'Tottenham Hotspur', 'Stockport':'Stockport County', 'Stoke':'Stoke City',
    'Swansea':'Swansea City', 'Tottenham':'Tottenham Hotspur', 'West Brom':'West Bromwich Albion',
    'West Ham':'West Ham United', 'Wigan':'Wigan Athletic', 'Wolves':'Wolverhampton Wanderers',
    'Wycombe':'Wycombe Wanderers',
}

def canonical(name):
    return normalise_team(NAMES.get(name, name), load_aliases())

def season_for(day):
    start = day.year - (day.month < 7)
    return f'{start}-{str(start+1)[-2:]}'

def supplement(matches, as_of):
    season = season_for(as_of)
    code = season[2:4] + season[-2:]
    rows, sources = [], []
    for year in range(int(season[:4])-2,int(season[:4])+1):
        source_season=f'{year}-{str(year+1)[-2:]}'; code=f'{year%100:02}{(year+1)%100:02}'
        for div, league in [('E0','premier-league'),('E1','championship'),('E2','league-one')]:
            url = f'https://www.football-data.co.uk/mmz4281/{code}/{div}.csv'
            raw = pd.read_csv(BytesIO(download(url)))
            sources.append({'url':url,'rows':len(raw),**provenance(url)})
            for _, r in raw.iterrows():
                day = pd.to_datetime(r.Date, dayfirst=True).date()
                if day >= as_of or season_for(day) != source_season: continue
                rows.append(dict(date=str(day), season=source_season, league=league, matchweek=None,
                    kickoff=r.get('Time'),home_team=canonical(r.HomeTeam),away_team=canonical(r.AwayTeam),
                    home_goals=r.FTHG,away_goals=r.FTAG,status='completed'))
    url = 'https://www.football-data.co.uk/fixtures.csv'
    raw = pd.read_csv(BytesIO(download(url))); sources.append({'url':url,'rows':len(raw),**provenance(url)})
    for _, r in raw[raw.Div.eq('E2')].iterrows():
        day = pd.to_datetime(r.Date, dayfirst=True).date()
        if day < as_of or season_for(day) != season: continue
        rows.append(dict(date=str(day),season=season,league='league-one',matchweek=None,kickoff=r.get('Time'),
            home_team=canonical(r.HomeTeam),away_team=canonical(r.AwayTeam),home_goals=None,away_goals=None,status='upcoming'))
    bootstrap = json.loads(download('https://fantasy.premierleague.com/api/bootstrap-static/'))
    teams = {t['id']:canonical(t['name']) for t in bootstrap['teams']}
    url = 'https://fantasy.premierleague.com/api/fixtures/'
    fixtures = json.loads(download(url)); sources.append({'url':url,'rows':len(fixtures),**provenance(url)})
    for f in fixtures:
        if not f['kickoff_time']: continue  # the coverage audit detects unscheduled fixtures
        stamp = pd.Timestamp(f['kickoff_time']).tz_convert('Europe/London')
        day = stamp.date()
        if season_for(day) != season: continue
        done = f['finished'] and f['team_h_score'] is not None and day < as_of
        rows.append(dict(date=str(day),season=season,league='premier-league',matchweek=f['event'],
            kickoff=stamp.strftime('%H:%M'),home_team=teams[f['team_h']],away_team=teams[f['team_a']],
            home_goals=f['team_h_score'] if done else None,away_goals=f['team_a_score'] if done else None,
            status='completed' if done else 'upcoming' if day >= as_of else 'incomplete'))
    extra = pd.DataFrame(rows)
    for side in ['home','away']: extra[side+'_team_id'] = extra[side+'_team'].map(team_id)
    keys=['season','league','home_team_id','away_team_id']
    # Retain round labels from the full schedule when the score feed omits them.
    old_weeks = matches.set_index(keys).matchweek
    extra['matchweek'] = extra.matchweek.fillna(pd.Series([old_weeks.get(tuple(r),None) for r in extra[keys].itertuples(index=False,name=None)],index=extra.index))
    result = pd.concat([matches,extra],ignore_index=True).drop_duplicates(keys,keep='last')
    return result.sort_values(['date','league','home_team']).reset_index(drop=True), sources
