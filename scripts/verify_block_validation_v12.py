"""Verify the frozen study's provenance and independently recompute its summaries."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.evaluation_validation import wilson_interval


def main():
    root = Path(__file__).resolve().parents[1]
    result = root/'results/block_validation_v12'
    manifest = json.loads((result/'manifest.json').read_text())
    completion = json.loads((result/'completion.json').read_text())
    config = json.loads((root/'configs/block_validation_v12.json').read_text())
    assert manifest['config'] == config
    paths = [root/'configs/block_validation_v12.json', root/config['design_source'],
        root/'scripts/run_block_validation.py', root/'src/openaffect_eeg/block_validation.py',
        root/'src/openaffect_eeg/exposure_statistics.py',
        root/'src/openaffect_eeg/evaluation_validation.py']
    assert {p.name: sha256_file(p) for p in paths} == manifest['sources']
    assert sha256_file(result/'replicates.csv') == completion['replicates_sha256']
    assert completion['failures'] == 0 and json.loads((result/'failures.json').read_text()) == []
    data = pd.read_csv(result/'replicates.csv')
    keys = ['dataset_id', 'scenario', 'method', 'estimand']
    summary = pd.read_csv(result/'summary.csv').set_index(keys)
    assert len(data) == 187200 and not data.duplicated(keys+['replicate']).any()
    assert len(data[['dataset_id', 'scenario', 'replicate']].drop_duplicates()) == 2400
    assert np.array_equal(data.covered, (data.ci_low-1e-12 <= data.truth)
        & (data.truth <= data.ci_high+1e-12))
    assert np.array_equal(data.positive, data.ci_low > 1e-12)
    for key, group in data.groupby(keys):
        row = summary.loc[key]
        assert len(group) == row.replicates == config['replicates']
        low, high = wilson_interval(int(group.covered.sum()), len(group))
        np.testing.assert_allclose(
            [row.coverage, row.coverage_mc_low, row.coverage_mc_high, row.bias,
             row.positive_rate, row.mean_width],
            [group.covered.mean(), low, high, (group.estimate-group.truth).mean(),
             group.positive.mean(), (group.ci_high-group.ci_low).mean()], atol=1e-12)
    print(json.dumps({'status': 'pass', 'simulation_replicates': 2400,
        'dependent_result_rows': len(data), 'summary_rows': len(summary),
        'boundary': 'Verifies records and arithmetic, not nominal inferential coverage.'}, indent=2))


if __name__ == '__main__':
    main()
