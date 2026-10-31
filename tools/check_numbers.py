"""Independent check of the main numbers of the notebook.

Recomputes them with plain Python and numpy only (no pandas, no scikit-learn) and compares
them with what the notebook wrote to results/. Run it from the repository folder after the
notebook:

    python tools/check_numbers.py

MohammadMahdi Faraji, mmyasharfaraji@gmail.com
"""
import ast
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
lines = []
failed = 0


def check(name, ours, notebook, tol=1e-3):
    global failed
    if isinstance(ours, (int, float)):
        ok = abs(ours - notebook) <= tol
    else:
        ok = ours == notebook
    failed += not ok
    lines.append(f'{"PASS" if ok else "FAIL"}  {name}: ours {ours}, notebook {notebook}')


def read_csv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


# the notebook's own summary: "name: value" per line
summary = {}
for line in (RESULTS / 'summary.txt').read_text(encoding='utf-8').splitlines():
    key, value = line.split(': ', 1)
    summary[key] = value

data = read_csv(ROOT / 'data' / 'synthetic_customer_behavior.csv')
category = [r['Purchase_Category'] for r in data]
known = [c for c in category if c != '']
classes = sorted(set(known))
counts = {c: known.count(c) for c in classes}

check('rows', len(data), int(summary['rows']))
check('missing categories', category.count(''), int(summary['missing categories']))
check('known counts', counts, ast.literal_eval(summary['known counts']))
majority = max(counts.values()) / len(known)
check('share of Electronics', round(majority, 4), float(summary['majority share']))

# correlations of the two strongest pairs
f = {i: np.array([float(r[f'Feature_{i}']) for r in data]) for i in range(1, 6)}
check('r Feature_2 Feature_5', round(np.corrcoef(f[2], f[5])[0, 1], 4), float(summary['r Feature_2 Feature_5']))
check('r Feature_1 Feature_4', round(np.corrcoef(f[1], f[4])[0, 1], 4), float(summary['r Feature_1 Feature_4']))

# spending of the top 20 %
amount = np.array([float(r['Purchase_Amount']) for r in data])
limit = np.quantile(amount, 0.8)
share = amount[amount >= limit].sum() / amount.sum()
check('top 20 % limit', round(limit, 2), float(summary['top 20 % limit']), tol=0.01)
check('top 20 % share of money', round(share, 4), float(summary['top 20 % share of money']))

# "always Electronics": accuracy = share of Electronics, F1 is 0 for the other three classes
comparison = {r['model']: r for r in read_csv(RESULTS / 'model_comparison.csv')}
dummy = comparison['always Electronics']
check('always Electronics accuracy', round(majority, 4), float(dummy['accuracy']))
precision, recall = majority, 1.0
f1_electronics = 2 * precision * recall / (precision + recall)
check('always Electronics macro F1', round(f1_electronics / 4, 4), float(dummy['macro F1']), tol=0.002)
check('always Electronics weighted F1', round(f1_electronics * majority, 4), float(dummy['weighted F1']), tol=0.002)
check('always Electronics balanced accuracy', 0.25, float(dummy['balanced accuracy']))

# the completed table: known rows unchanged, 700 rows filled and marked, same counts as the notebook
completed = read_csv(RESULTS / 'completed_dataset.csv')
check('completed rows', len(completed), len(data))
unchanged = all(c['Purchase_Category'] == d['Purchase_Category'] for c, d in zip(completed, data) if d['Purchase_Category'])
check('known categories unchanged', unchanged, True)
marked = [c['category_filled'] == 'True' for c in completed]
check('filled rows marked', sum(marked), category.count(''))
check('every row has a category', all(c['Purchase_Category'] for c in completed), True)
filled = [c['Purchase_Category'] for c, m in zip(completed, marked) if m]
check('filled counts', {c: filled.count(c) for c in classes if filled.count(c)}, ast.literal_eval(summary['filled by random draws']))

lines.append(f'\n{len(lines) - failed} of {len(lines)} checks passed')
text = '\n'.join(lines)
print(text)
(RESULTS / 'check_numbers.txt').write_text(text + '\n', encoding='utf-8', newline='\n')
