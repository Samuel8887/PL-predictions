"""Descriptive paired date-cluster bootstrap, not a new model-selection step."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'site/data/evaluation_fixtures.json').read_text(encoding='utf8'))
if not rows or rows[0]['original'] is None: raise SystemExit('Original comparison unavailable')
groups={}
for row in rows:
    y=row['outcome']
    groups.setdefault(row['date'],[]).append(-np.log(row['corrected'][y])+np.log(row['original'][y]))
values=list(groups.values()); totals=np.array([sum(g) for g in values]);counts=np.array([len(g) for g in values])
rng=np.random.default_rng(20260916);samples=[]
for _ in range(3000):
    indexes=rng.integers(0,len(values),len(values));samples.append(totals[indexes].sum()/counts[indexes].sum())
result={'comparison':'corrected minus original log loss; negative favors corrected','mean_difference':float(totals.sum()/counts.sum()),
    'bootstrap_unit':'match date','date_clusters':len(values),'resamples':3000,'seed':20260916,
    'interval_95':np.quantile(samples,[.025,.975]).tolist(),'warning':'Descriptive single-season comparison; fixtures and teams are dependent.'}
(ROOT/'artifacts').mkdir(exist_ok=True)
(ROOT/'artifacts/comparison_uncertainty.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
