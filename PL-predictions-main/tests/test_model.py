"""Offline regression tests for probability, leakage and player correctness."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_predict as model
from import_player_data import player_key
from import_data import parse_day, parse_fixture


def matches(goals=(2, 1)):
    return pd.DataFrame([dict(date=pd.Timestamp('2025-01-01') + pd.Timedelta(days=i),
        league='premier-league', season='2024-25', home_team='Home', away_team='Away',
        home_team_id='home', away_team_id='away', home_goals=goals[0], away_goals=goals[1]) for i in range(60)])


def squad():
    return pd.DataFrame([dict(player=f'Player {i}', player_key=f'player{i}', position=pos,
        model_team='Home', status='a', chance_of_playing=100, starts=4, minutes=321+i,
        goals=1, assists=1, xg=.6789+i/10, xa=.4321, start_probability=.912345)
        for i,pos in enumerate(['GKP']*2+['DEF']*7+['MID']*7+['FWD']*4)])


class ModelTests(unittest.TestCase):
    def test_high_scoring_tail_is_normalized(self):
        self.assertAlmostEqual(sum(model.probability(15, 12)), 1, places=12)

    def test_calibrated_score_grid_matches_outcomes(self):
        grid = model.score_distribution(2.1, .8, 1.7)
        actual = [np.tril(grid,-1).sum(), np.trace(grid), np.triu(grid,1).sum()]
        np.testing.assert_allclose(actual, model.sharpen(model.probability(2.1,.8),1.7))

    def test_invalid_goal_rates(self):
        for value in [-1, np.nan, np.inf]:
            with self.assertRaises(ValueError): model.probability(value, 1)

    def test_zero_goals(self):
        self.assertEqual(model.probability(0,0), (0.,1.,0.))
        history = matches((0,0))
        fixture = history.iloc[-1].copy(); fixture.date += pd.Timedelta(days=1)
        self.assertEqual(model.state_prediction(history, fixture)['draw'], 1)

    def test_same_day_and_future_results_excluded(self):
        history=matches(); fixture=history.iloc[-1].copy(); fixture.date+=pd.Timedelta(days=1)
        original=model.state_prediction(history,fixture)
        extra=fixture.copy(); extra.home_goals=1000
        self.assertEqual(original, model.state_prediction(pd.concat([history,pd.DataFrame([extra])]),fixture))

    def test_tied_season_probabilities_sum_to_one(self):
        played=matches((0,0)); future=played.iloc[:1].copy(); future.date=pd.Timestamp('2025-05-01')
        pred={'expected_home_goals':0.,'expected_away_goals':0.}
        result=model.season_outlook(played,future,'premier-league',predictions={0:pred},simulations=20)
        for row in result['table']:
            self.assertEqual(row['position_probabilities'], [.5,.5])
            self.assertEqual(row['expected_position'],1.5)
        self.assertEqual(sum(row['win_probability'] for row in result['table']),1)

    def test_simulation_uses_calibration(self):
        played=matches(); future=played.iloc[:1].copy()
        with patch.object(model,'score_distribution', wraps=model.score_distribution) as score:
            model.season_outlook(played,future,'premier-league',1.7,predictions={0:{'expected_home_goals':2.,'expected_away_goals':1.}},simulations=20)
            score.assert_called_once_with(2.,1.,1.7)

    def test_lineup_has_legal_formation_and_no_duplicate_bench(self):
        lineup,bench=model.projected_squad(squad())
        self.assertEqual(len(lineup),11)
        self.assertEqual(sum(lineup.position.eq('GKP')),1)
        self.assertTrue(3 <= sum(lineup.position.eq('DEF')) <= 5)
        self.assertEqual(len(set(lineup.index)&set(bench.index)),0)

    def test_player_metrics_survive_fractional_join_and_empty_priors(self):
        players=squad(); players.loc[0,'status']='i'
        result=model.player_predictions(players,pd.DataFrame(),pd.Series({'home_team':'Home','away_team':'Away'}),2.,1.)[0]
        self.assertNotIn('Player 0',[p['player'] for p in result['projected_lineup']+result['projected_bench']])
        self.assertTrue(all(p['score_probability']>0 for p in result['projected_lineup'] if p['position']!='GKP'))

    def test_historical_prior_has_correct_units(self):
        players=squad(); players[['xg','goals','xa','assists']]=0
        # Equal per-minute priors must produce equal contributions for equal minutes.
        players['minutes']=300
        priors=pd.DataFrame({'player_key':players.player_key, 'prior_minutes':900.,'prior_xg':9.,'prior_xa':9.,'prior_goals':0.,'prior_assists':0.})
        result=model.player_predictions(players,priors,pd.Series({'home_team':'Home','away_team':'Away'}),2.,1.)[0]
        # A known historical rate and the same positional default must be equivalent.
        priors['prior_xg']=players.position.map({'GKP':0,'DEF':.6,'MID':1.6,'FWD':2.8})
        with_prior=model.player_predictions(players,priors,pd.Series({'home_team':'Home','away_team':'Away'}),2.,1.)[0]
        without=model.player_predictions(players,pd.DataFrame(),pd.Series({'home_team':'Home','away_team':'Away'}),2.,1.)[0]
        self.assertAlmostEqual(with_prior['top_scorer']['score_probability'],without['top_scorer']['score_probability'])

    def test_archive_player_key(self):
        self.assertEqual(player_key('José_Sá_123'),player_key('José Sá'))

    def test_parser_formats_and_leap_day(self):
        self.assertEqual(str(parse_day('Sat Feb 29','2019-20')),'2020-02-29')
        self.assertEqual(parse_fixture('Home 2-1 (1-0) Away'),('Home','Away',None,2,1))
        self.assertEqual(parse_fixture('15:00 Home v Away 2-1'),('Home','Away','15:00',2,1))

if __name__=='__main__': unittest.main()
