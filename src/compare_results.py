"""Compare freshly generated result JSON files with the expected ones used in the paper.

Usage: python3 compare_results.py <new_results_dir> <expected_dir>
Numbers are compared with a small tolerance; timing fields are ignored.
"""
import sys, os, json

new_dir, exp_dir = sys.argv[1], sys.argv[2]
IGNORE = ('fit_seconds', 'seconds', 'time')
TOL = 1e-9
problems = 0


def cmp(a, b, path):
    global problems
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if any(t in k for t in IGNORE):
                continue
            if k not in a or k not in b:
                problems += 1; print(f'  missing key: {path}/{k}')
            else:
                cmp(a[k], b[k], f'{path}/{k}')
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems += 1; print(f'  length differs: {path}')
        for i, (x, y) in enumerate(zip(a, b)):
            cmp(x, y, f'{path}[{i}]')
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > TOL * max(1.0, abs(b)):
            problems += 1; print(f'  differs: {path}: new {a} vs expected {b}')
    elif a != b:
        problems += 1; print(f'  differs: {path}: new {a!r} vs expected {b!r}')


for f in sorted(os.listdir(exp_dir)):
    if not f.endswith('.json'):
        continue
    p = os.path.join(new_dir, f)
    if not os.path.exists(p):
        problems += 1; print(f'{f}: not produced'); continue
    before = problems
    cmp(json.load(open(p)), json.load(open(os.path.join(exp_dir, f))), f)
    print(f'{f}: {"identical" if problems == before else "DIFFERENCES above"}')

print('All results reproduced exactly.' if problems == 0 else f'{problems} difference(s) found.')
sys.exit(1 if problems else 0)
