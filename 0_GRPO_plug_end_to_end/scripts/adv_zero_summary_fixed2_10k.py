import glob
import hashlib
import json
import re

import numpy as np
import pandas as pd

rows = []
files = sorted(glob.glob('rollout_data_batch_fixed_2_10k/*.jsonl'))
if not files:
    raise SystemExit('No jsonl files found in rollout_data_batch_fixed_2_10k')

for f in files:
    groups = {}
    with open(f, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            g = hashlib.md5(r['input'].encode()).hexdigest()
            groups.setdefault(g, []).append(float(r.get('score', np.nan)))

    m = re.search(r'/([0-9]+)\.jsonl$', f)
    step = int(m.group(1)) if m else -1
    for g, scores in groups.items():
        rv = float(np.var(scores))
        rows.append(
            {
                'file': f,
                'step': step,
                'prompt_group': g,
                'group_size': len(scores),
                'reward_var': rv,
                'adv_zero_group': rv <= 1e-12,
            }
        )

df = pd.DataFrame(rows)
summary = (
    df.groupby('step', as_index=False)
    .agg(
        groups=('prompt_group', 'count'),
        group_size_mean=('group_size', 'mean'),
        adv_zero_groups=('adv_zero_group', 'sum'),
        adv_zero_ratio=('adv_zero_group', 'mean'),
        reward_var_mean=('reward_var', 'mean'),
    )
    .sort_values('step')
)

out1 = 'rollout_data_batch_fixed_2_10k/group_stats_with_adv_zero_flag.csv'
out2 = 'rollout_data_batch_fixed_2_10k/step_summary_adv_zero.csv'
df.to_csv(out1, index=False)
summary.to_csv(out2, index=False)

print('files=', len(files))
print('saved detail:', out1)
print('saved summary:', out2)
print('overall_adv_zero_ratio_mean=', summary['adv_zero_ratio'].mean())
print('overall_adv_zero_ratio_min =', summary['adv_zero_ratio'].min())
print('overall_adv_zero_ratio_max =', summary['adv_zero_ratio'].max())
print('\nfirst 10 steps:')
print(summary.head(10).to_string(index=False))
print('\nlast 10 steps:')
print(summary.tail(10).to_string(index=False))
