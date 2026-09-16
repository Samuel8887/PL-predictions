"""Exercise the entire JSON build against an isolated, synthetic dataset."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_predict as model

class BuildTests(unittest.TestCase):
    def test_offline_build(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root=Path(temp)
            (root/'data').mkdir(); (root/'config').mkdir()
            (root/'config/team_colours.json').write_text('{}')
            rows=[]
            for i in range(100):
                rows.append(dict(date=(pd.Timestamp('2025-01-01')+pd.Timedelta(days=i)).date(),season='2026-27',league='premier-league',home_team='Home',away_team='Away',home_team_id='home',away_team_id='away',home_goals=i%4,away_goals=i%3,status='completed',matchweek=i+1,kickoff=None))
            for i in range(2):
                rows.append(dict(rows[-1],date=pd.Timestamp.now().normalize()+pd.Timedelta(days=i+1),home_team=f'New club {i}',home_team_id=f'new-club-{i}',home_goals=None,away_goals=None,status='upcoming',matchweek=101+i))
            pd.DataFrame(rows).to_csv(root/'data/matches.csv',index=False)
            with patch.object(model,'ROOT',root),patch.object(model,'OUT',root/'site/data'):
                model.main(compare=False)
            data=json.loads((root/'site/data/predictions.json').read_text())
            self.assertEqual(data['model_version'],'2.1')
            self.assertEqual(len(data['leagues']['premier-league']),2)
            self.assertNotIn('premier-league',data['season_outlook'])
            self.assertFalse(data['coverage']['premier-league']['schedule_complete'])
