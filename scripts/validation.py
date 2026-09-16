"""Coverage and artifact invariants shared by local and CI builds."""
import argparse
import json
import math
from pathlib import Path
import pandas as pd

LEAGUE_SIZES={'premier-league':20,'championship':24,'league-one':24}

def coverage(matches,season):
    report={}
    for league,n in LEAGUE_SIZES.items():
        rows=matches[matches.league.eq(league)&matches.season.eq(season)]
        teams=set(rows.home_team)|set(rows.away_team)
        pairs=set(zip(rows.home_team,rows.away_team))
        duplicates=len(rows)-len(pairs)
        unresolved=int(rows.status.eq('incomplete').sum())
        complete=len(teams)==n and len(pairs)==n*(n-1) and duplicates==0 and all(a!=b for a,b in pairs)
        reasons=[]
        if not complete: reasons.append(f'Partial schedule: {len(pairs)} of {n*(n-1)} expected regular-season fixtures.')
        if unresolved: reasons.append(f'{unresolved} past fixtures have no confirmed result.')
        if league=='league-one': reasons.append('openfootball has no current-season League One file; only results and forthcoming fixtures available from football-data.co.uk are included.')
        report[league]=dict(season=season,teams=len(teams),expected_teams=n,fixtures=len(rows),expected_fixtures=n*(n-1),
            completed=int(rows.status.eq('completed').sum()),upcoming=int(rows.status.eq('upcoming').sum()),unresolved=unresolved,
            duplicate_pairs=duplicates,schedule_complete=complete,season_forecast_available=complete and unresolved==0,
            notice=' '.join(reasons))
    return report

def validate_predictions(data):
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value): raise ValueError('Non-finite JSON value')
        if isinstance(value,dict):
            for v in value.values(): finite(v)
        elif isinstance(value,list):
            for v in value: finite(v)
    finite(data)
    as_of=data.get('as_of',data['generated_at'][:10])
    for league,fixtures in data['leagues'].items():
        seen=set()
        for f in fixtures:
            key=(f['home_team'],f['away_team'])
            if key in seen: raise ValueError('Duplicate fixture')
            seen.add(key)
            if f['date']<as_of: raise ValueError('Past fixture presented as upcoming')
            if f['prediction_available']:
                probs=[f[k] for k in ['home_win','draw','away_win']]
                if any(p<0 or p>1 for p in probs) or abs(sum(probs)-1)>0.0002: raise ValueError('Invalid match probabilities')
            for side in f.get('player_predictions',[]):
                lineup=side['projected_lineup']; bench=side['projected_bench']
                if lineup:
                    counts={p:sum(r['position']==p for r in lineup) for p in ['GKP','DEF','MID','FWD']}
                    if len(lineup)!=11 or counts['GKP']!=1 or (counts['DEF'],counts['MID'],counts['FWD']) not in [(4,4,2),(4,3,3),(4,5,1),(3,5,2),(3,4,3),(5,3,2),(5,4,1)]: raise ValueError('Illegal starting formation')
                if len(bench)>9 or len({p['player'] for p in lineup+bench})!=len(lineup+bench): raise ValueError('Duplicate squad member')
                for player in lineup+bench:
                    for key in ['start_probability','score_probability','assist_probability']:
                        if not 0<=player[key]<=1: raise ValueError('Invalid player probability')
    for league,outlook in data['season_outlook'].items():
        if not outlook: continue
        table=outlook['table']; n=len(table)
        if not data['coverage'][league]['season_forecast_available']: raise ValueError('Incomplete schedule used for season simulation')
        for r in table:
            if len(r['position_probabilities'])!=n or abs(sum(r['position_probabilities'])-1)>.002: raise ValueError('Invalid finishing distribution')
        for i in range(n):
            if abs(sum(r['position_probabilities'][i] for r in table)-1)>.002: raise ValueError('Invalid position mass')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,default=Path('site'));args=parser.parse_args()
    data=json.loads((args.site/'data/predictions.json').read_text(encoding='utf8'));validate_predictions(data)
    for file in ['evaluation.json','team_colours.json','player_report.json','data_quality.json']:
        json.loads((args.site/'data'/file).read_text(encoding='utf8'))
    for file in ['index.html','app.js','styles.css']:
        if not (args.site/file).is_file(): raise ValueError(f'Missing asset: {file}')
    print('Forecast invariants, strict JSON and Pages artifact files verified.')
