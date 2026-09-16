"""Verify real imported identities, schedules and published player eligibility."""
import json
from pathlib import Path
import pandas as pd
from current_matches import canonical
from validation import validate_predictions

ROOT=Path(__file__).resolve().parents[1]

def main():
    matches=pd.read_csv(ROOT/'data/matches.csv')
    current=pd.read_csv(ROOT/'data/current_players.csv')
    history=pd.read_csv(ROOT/'data/player_history.csv')
    predictions=json.loads((ROOT/'site/data/predictions.json').read_text(encoding='utf8'))
    validate_predictions(predictions)
    assert not matches.duplicated(['season','league','home_team_id','away_team_id']).any()
    assert not current.player_key.duplicated().any()
    assert not history.duplicated(['season','player_key']).any()
    assert not matches.home_team.eq(matches.away_team).any()
    for side in ['home','away']:
        assert not matches[side+'_team'].str.contains(r'\[|\]|a\.e\.t\.|pen\.|\d-\d',regex=True).any()
    dates=pd.to_datetime(matches.date)
    assert (dates[matches.status.eq('completed')]<pd.Timestamp(predictions['as_of'])).all()
    assert matches.loc[matches.status.eq('completed'),['home_goals','away_goals']].notna().all().all()
    assert (matches.loc[matches.status.eq('completed'),['home_goals','away_goals']]>=0).all().all()
    for season, group in matches.groupby('season'):
        year=int(season[:4])
        assert (group.date >= f'{year}-07-01').all()
        end=f'{year+1}-08-31' if season=='2019-20' else f'{year+1}-07-31'
        assert (group.date <= end).all(),season
    current['model_team']=current.team.map(canonical)
    checked=0
    for fixtures in predictions['leagues'].values():
        for fixture in fixtures:
            for side in fixture.get('player_predictions',[]):
                assert pd.Timestamp(fixture['date'])<=pd.Timestamp(predictions['as_of'])+pd.Timedelta(days=14)
                for player in side['projected_lineup']+side['projected_bench']:
                    rows=current[current.model_team.eq(side['team'])&current.player.eq(player['player'])]
                    assert len(rows)==1
                    row=rows.iloc[0];assert row.status in ['a','d'] and row.chance_of_playing>0
                    assert player['score_probability']<=row.chance_of_playing/100+.0001
                    assert player['assist_probability']<=row.chance_of_playing/100+.0001
                    checked+=1
    summary={'matches':len(matches),'completed':int(matches.status.eq('completed').sum()),
        'match_date_range':[matches.date.min(),matches.date.max()], 'player_seasons':len(history),'current_players':len(current),
        'published_player_rows_checked':checked,'duplicate_fixture_pairs':0,'duplicate_current_player_codes':0,
        'checks':'Fixture identities, dates, same-day cutoff, source annotations, persistent player identity, actual availability, probability bounds and all output invariants passed.',
        'coverage':predictions['coverage']}
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/source_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
