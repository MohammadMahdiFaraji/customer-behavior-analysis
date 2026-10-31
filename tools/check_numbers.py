"""Independent check of the main numbers of the notebook.

It computes them again with plain Python and numpy (no pandas, no scikit-learn) and compares
them with the files that the notebook wrote to results/. Run it from the repository folder
after running the notebook:

    python tools/check_numbers.py

MohammadMahdi Faraji, mmyasharfaraji@gmail.com
"""
import ast
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
lines = []     # one line of text for every check
results = []   # True or False for every check


def check(name, ours, notebook, tol=1e-3):
    # numbers are compared with a small tolerance, everything else must be equal
    if isinstance(ours, (int, float)):
        ok = abs(ours - notebook) <= tol
    else:
        ok = ours == notebook
    results.append(ok)
    if ok:
        status = 'PASS'
    else:
        status = 'FAIL'
    lines.append(f'{status}  {name}: ours {ours}, notebook {notebook}')


def read_csv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


# the summary of the notebook: one "name: value" per line
summary = {}
for line in (RESULTS / 'summary.txt').read_text(encoding='utf-8').splitlines():
    key, value = line.split(': ', 1)
    summary[key] = value

# the data, and the categories of the 300 known customers
data = read_csv(ROOT / 'data' / 'synthetic_customer_behavior.csv')
category = [row['Purchase_Category'] for row in data]
known = [c for c in category if c != '']
classes = sorted(set(known))
counts = {}
for c in classes:
    counts[c] = known.count(c)

check('rows', len(data), int(summary['rows']))
check('missing categories', category.count(''), int(summary['missing categories']))
check('known counts', counts, ast.literal_eval(summary['known counts']))
majority = max(counts.values()) / len(known)
check('share of Electronics', round(majority, 4), float(summary['majority share']))

# correlations of the two strongest pairs
features = {}
for i in range(1, 6):
    features[i] = np.array([float(row['Feature_' + str(i)]) for row in data])
r25 = np.corrcoef(features[2], features[5])[0, 1]
r14 = np.corrcoef(features[1], features[4])[0, 1]
check('r Feature_2 Feature_5', round(r25, 4), float(summary['r Feature_2 Feature_5']))
check('r Feature_1 Feature_4', round(r14, 4), float(summary['r Feature_1 Feature_4']))

# spending of the top 20 %
amount = np.array([float(row['Purchase_Amount']) for row in data])
limit = np.quantile(amount, 0.8)
share = amount[amount >= limit].sum() / amount.sum()
check('top 20 % limit', round(limit, 2), float(summary['top 20 % limit']), tol=0.01)
check('top 20 % share of money', round(share, 4), float(summary['top 20 % share of money']))

# "always Electronics": accuracy = share of Electronics, F1 is 0 for the other three classes
comparison = {}
for row in read_csv(RESULTS / 'model_comparison.csv'):
    comparison[row['model']] = row
dummy = comparison['always Electronics']
check('always Electronics accuracy', round(majority, 4), float(dummy['accuracy']))
precision = majority   # it predicts Electronics for everyone
recall = 1.0           # and so it finds every Electronics buyer
f1_electronics = 2 * precision * recall / (precision + recall)
check('always Electronics macro F1', round(f1_electronics / 4, 4), float(dummy['macro F1']), tol=0.002)
check('always Electronics weighted F1', round(f1_electronics * majority, 4), float(dummy['weighted F1']), tol=0.002)
check('always Electronics balanced accuracy', 0.25, float(dummy['balanced accuracy']))

# the completed table: known rows unchanged, 700 rows filled and marked, same counts as the notebook
completed = read_csv(RESULTS / 'completed_dataset.csv')
check('completed rows', len(completed), len(data))

unchanged = True
for c, d in zip(completed, data):
    if d['Purchase_Category'] != '' and c['Purchase_Category'] != d['Purchase_Category']:
        unchanged = False
check('known categories unchanged', unchanged, True)

marked = [c['category_filled'] == 'True' for c in completed]
check('filled rows marked', sum(marked), category.count(''))

every_row = True
for c in completed:
    if c['Purchase_Category'] == '':
        every_row = False
check('every row has a category', every_row, True)

filled = []
for c, m in zip(completed, marked):
    if m:
        filled.append(c['Purchase_Category'])
filled_counts = {}
for c in classes:
    if filled.count(c) > 0:
        filled_counts[c] = filled.count(c)
check('filled counts', filled_counts, ast.literal_eval(summary['filled by random draws']))

lines.append(f'\n{sum(results)} of {len(results)} checks passed')
text = '\n'.join(lines)
print(text)
(RESULTS / 'check_numbers.txt').write_text(text + '\n', encoding='utf-8', newline='\n')
