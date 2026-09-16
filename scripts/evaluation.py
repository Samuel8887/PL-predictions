"""Fixed seasonal, rolling-origin evaluation; only earlier validation tunes calibration."""
from datetime import date
import importlib.util
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

LEGACY_REVISION = '45ad019'

def original_model(root):
    result = subprocess.run(['git','show',f'{LEGACY_REVISION}:scripts/train_predict.py'],cwd=root,capture_output=True,text=True,encoding='utf8')
    if result.returncode: return None
    path = root / '.cache' / 'original_model.py'
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(result.stdout,encoding='utf8')
    spec=importlib.util.spec_from_file_location('original_model',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def vector(pred): return [pred['home_win'],pred['draw'],pred['away_win']]

def metrics(y, probabilities):
    y=np.asarray(y,dtype=int); p=np.asarray(probabilities,dtype=float)
    confidence=p.max(axis=1); correct=p.argmax(axis=1)==y
    bins=[]
    for lo,hi in zip([0,.2,.4,.6,.8],[.2,.4,.6,.8,1.0000001]):
        mask=(confidence>=lo)&(confidence<hi)
        if mask.any(): bins.append(dict(range=f'{lo:.1f}-{min(hi,1):.1f}',n=int(mask.sum()),mean_confidence=float(confidence[mask].mean()),accuracy=float(correct[mask].mean())))
    return dict(fixtures=len(y),accuracy=float(correct.mean()),log_loss=float(-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean()),
        brier_score=float(((p-np.eye(3)[y])**2).sum(axis=1).mean()),calibration=bins,
        expected_calibration_error=sum(b['n']*abs(b['mean_confidence']-b['accuracy']) for b in bins)/len(y))

def tune(model,history,validation):
    rows=[]
    # Deterministic sample from validation only. Final test is never used to select parameters.
    for _,fixture in validation.iloc[::max(1,int(np.ceil(len(validation)/500)))].iterrows():
        pred=model.state_prediction(history,fixture)
        if pred: rows.append((model.outcome(fixture.home_goals,fixture.away_goals),vector(pred)))
    if not rows: return 1.,0
    y=np.array([r[0] for r in rows]); p=np.array([r[1] for r in rows])
    candidates=np.round(np.arange(.8,2.01,.1),1)
    losses=[]
    for value in candidates:
        calibrated=p**value; calibrated/=calibrated.sum(axis=1,keepdims=True)
        losses.append(-np.log(calibrated[np.arange(len(y)),y].clip(1e-12)).mean())
    return float(candidates[np.argmin(losses)]),len(y)

def evaluate_models(completed,model,root,test_start=None,test_end=None,compare=True):
    today=date.today(); current_year=today.year-(today.month<7)
    start=pd.Timestamp(test_start or f'{current_year-1}-07-01')
    end=pd.Timestamp(test_end or f'{current_year}-07-01')
    validation_start=start-pd.DateOffset(years=1)
    history=completed.sort_values(['date','league','home_team_id']).reset_index(drop=True)
    validation=history[(history.date>=validation_start)&(history.date<start)]
    alpha,nval=tune(model,history[history.date<start],validation)
    old=original_model(root) if compare else None
    old_alpha,old_nval=tune(old,history[history.date<start],validation) if old else (None,0)
    test=history[(history.date>=start)&(history.date<end)]
    rows=[]; skipped=0
    for _,fixture in test.iterrows():
        pred=model.state_prediction(history,fixture,alpha)
        previous=old.state_prediction(history,fixture,old_alpha) if old else None
        if not pred or (old and not previous): skipped+=1; continue
        baseline=history[(history.date<start)&history.league.eq(fixture.league)]
        counts=np.array([(baseline.home_goals>baseline.away_goals).sum(),(baseline.home_goals==baseline.away_goals).sum(),(baseline.home_goals<baseline.away_goals).sum()],dtype=float)+1
        base=counts/counts.sum()
        rows.append(dict(date=str(fixture.date.date()),league=fixture.league,home_team=fixture.home_team,away_team=fixture.away_team,
            outcome=model.outcome(fixture.home_goals,fixture.away_goals),corrected=vector(pred),baseline=base.tolist(),original=vector(previous) if old else None))
    if not rows: return dict(message='No eligible fixtures in fixed final-test period.',probability_sharpness=alpha),[]
    def summarize(group):
        y=[r['outcome'] for r in group]
        result={key:metrics(y,[r[key] for r in group]) for key in ['corrected','baseline']+(['original'] if old else [])}
        result.update(date_start=min(r['date'] for r in group),date_end=max(r['date'] for r in group))
        return result
    total=summarize(rows)
    report=dict(validation='Full final completed season; rolling origin, strictly earlier dates only.',
        test_start=str(start.date()),test_end_exclusive=str(end.date()),validation_start=str(validation_start.date()),
        validation_end_exclusive=str(start.date()),validation_fixtures=nval,eligible_test_fixtures=len(test),skipped_test_fixtures=skipped,
        probability_sharpness=alpha,original_probability_sharpness=old_alpha,original_validation_fixtures=old_nval,
        original_revision=LEGACY_REVISION if old else None,
        comparison='Identical repaired inputs, test fixtures and information cutoffs; only model code differs.' if old else 'Direct comparison unavailable: original Git revision not accessible.',
        baseline_definition='Per-league historical H/D/A frequencies before test start, add-one smoothed; frozen throughout test.',
        player_evaluation='Not performed. Current player snapshots never enter this match backtest.',
        overall=total,per_league={league:summarize([r for r in rows if r['league']==league]) for league in sorted(set(r['league'] for r in rows))},
        **total['corrected'],baseline_accuracy=total['baseline']['accuracy'],baseline_log_loss=total['baseline']['log_loss'],baseline_brier_score=total['baseline']['brier_score'])
    return report,rows
