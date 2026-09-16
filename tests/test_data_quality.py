import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from import_data import parse_repo,parse_fixture
from current_matches import canonical
from validation import coverage
from evaluation import metrics,tune
import train_predict as model
from test_model import squad,matches

class DataQualityTests(unittest.TestCase):
    def test_covid_summer_dates_use_season_end_year(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'2019-20';folder.mkdir()
            (folder/'1-premierleague.txt').write_text('# Date Fri Aug 9 2019 - Sun Jul 26 2020 (352d)\nSun Jul 26\nHome 2-1 Away',encoding='utf8')
            row=parse_repo(Path(temp),date(2026,9,16)).iloc[0]
            self.assertEqual(row.date,'2020-07-26')

    def test_rounds_annotations_rescheduling_and_playoffs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'2025-26';folder.mkdir()
            (folder/'2-championship.txt').write_text('''▪ Regular Season - 1
Sat Aug 2 2025
  15:00 Home FC v Away FC [postponed]
▪ 2. Round
Sat Aug 9
  15:00 Home FC v Away FC 2-1
         Other FC v Third FC 0-0
▪ Finals, Final
Sat May 30 2026
  Home FC 4-3 pen. 1-1 a.e.t. Away FC
''',encoding='utf8')
            result=parse_repo(Path(temp),date(2026,9,16))
            self.assertEqual(len(result),2)
            self.assertEqual(set(result.matchweek),{2})
            self.assertEqual(set(result.kickoff),{'15:00'})
            self.assertEqual(set(result.home_team),{'Home','Other'})

    def test_same_day_source_score_is_not_training_data(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'2025-26';folder.mkdir()
            (folder/'1-premierleague.txt').write_text('Sat Aug 2 2025\n15:00 Home v Away 9-0',encoding='utf8')
            row=parse_repo(Path(temp),date(2025,8,2)).iloc[0]
            self.assertEqual(row.status,'upcoming');self.assertTrue(pd.isna(row.home_goals))

    def test_supplementary_identity_aliases(self):
        for short,long in [('Bristol Rvs','Bristol Rovers'),('Man United','Manchester United'),('Northampton','Northampton Town'),('Nott\'ham Forest','Nottingham Forest')]:
            self.assertEqual(canonical(short),long)

    def test_incomplete_or_duplicate_schedule_blocks_table(self):
        teams=[f'T{i}' for i in range(20)]
        rows=[dict(league='premier-league',season='2026-27',home_team=a,away_team=b,status='upcoming') for a in teams for b in teams if a!=b]
        frame=pd.DataFrame(rows)
        self.assertTrue(coverage(frame,'2026-27')['premier-league']['season_forecast_available'])
        self.assertFalse(coverage(frame.iloc[:-1],'2026-27')['premier-league']['season_forecast_available'])
        frame.loc[0,'status']='incomplete'
        self.assertFalse(coverage(frame,'2026-27')['premier-league']['season_forecast_available'])
        self.assertFalse(coverage(pd.concat([frame,frame.iloc[:1]]),'2026-27')['premier-league']['season_forecast_available'])

    def test_missing_xg_years_do_not_dilute_observed_xg(self):
        history=pd.DataFrame([dict(player_key='p',season='2025-26',minutes=900,xg=9,xa=4,goals=8,assists=4,xg_minutes=900,xa_minutes=900),
            dict(player_key='p',season='2021-22',minutes=3000,xg=0,xa=0,goals=20,assists=9,xg_minutes=0,xa_minutes=0)])
        prior=model.player_priors(history).iloc[0]
        self.assertAlmostEqual(prior.prior_xg/prior.prior_xg_minutes,.01)

    def test_zero_player_rates_finite_and_availability_caps(self):
        current=squad();current[['xg','xa','goals','assists']]=0;current['chance_of_playing']=25
        prior=pd.DataFrame(dict(player_key=current.player_key,prior_minutes=900,prior_xg=0,prior_xa=0,prior_goals=0,prior_assists=0))
        side=model.player_predictions(current,prior,pd.Series(dict(home_team='Home',away_team='Away')),8,1)[0]
        players=side['projected_lineup']+side['projected_bench']
        for p in players:
            self.assertTrue(np.isfinite(p['score_probability']))
            self.assertLessEqual(p['score_probability'],.25)
            self.assertLessEqual(p['assist_probability'],.25)
        self.assertLessEqual(sum(p['start_probability'] for p in players),11.001)

    def test_future_and_unfinished_rows_cannot_change_prediction(self):
        history=matches();fixture=history.iloc[-1].copy();fixture.date+=pd.Timedelta(days=1)
        original=model.state_prediction(history,fixture)
        extra=fixture.copy();extra.date+=pd.Timedelta(days=10);extra.home_goals=1000
        future=pd.concat([history,pd.DataFrame([extra])],ignore_index=True)
        self.assertEqual(original,model.state_prediction(future,fixture))
        future['status']='completed';future.loc[len(future)-1,'date']=fixture.date-pd.Timedelta(days=1);future.loc[len(future)-1,'status']='incomplete'
        self.assertEqual(original,model.state_prediction(future,fixture))

    def test_calibration_includes_confidence_one(self):
        result=metrics([0,1],[[1,0,0],[0,1,0]])
        self.assertEqual(sum(b['n'] for b in result['calibration']),2)
        self.assertEqual(result['brier_score'],0)

    def test_validation_tuning_excludes_final_test(self):
        history=matches(); validation=history.iloc[-10:]
        with patch.object(model,'state_prediction',wraps=model.state_prediction) as predict:
            first=tune(model,history,validation)
            self.assertEqual(len(predict.call_args_list),10)
            self.assertTrue(all(call.args[1].date<=validation.date.max() for call in predict.call_args_list))
        future=history.iloc[-1].copy();future.date=pd.Timestamp('2030-01-01');future.home_goals=1000
        self.assertEqual(first,tune(model,pd.concat([history,pd.DataFrame([future])]),validation))

if __name__=='__main__':unittest.main()
