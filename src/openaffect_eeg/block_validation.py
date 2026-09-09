"""Validate the actual mean-of-blocks estimand, retaining fixed-fit boundaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openaffect_eeg.evaluation_validation import simulate_oracle
from openaffect_eeg.exposure_statistics import weighted_scores


def simulate_blocks(*, supports, seed, scenario, noise_sd, doses, target_names):
    """Complete diagonal test blocks with independent synthetic resource pools.

    Matches test topology, not the shared fitted models/resource graph of EEG data.
    Refuses irregular topology instead of silently changing the requested support.
    """
    parts = []
    truth = None
    for block, support in enumerate(supports):
        n = support['n_participants']
        m = support['n_stimuli_or_conditions']
        if support['n_trials'] != n*m:
            raise ValueError('This generator requires complete diagonal blocks')
        table, exact = simulate_oracle(
            seed=seed+block*100003, participants=n, stimuli=m,
            density=1.0, noise_sd=noise_sd, scenario=scenario,
            participant_doses=doses, stimulus_doses=doses)
        table['seed'] = block
        for col in ('trial_uid', 'subject_uid', 'stimulus_uid'):
            table[col] = str(block)+':'+table[col].astype(str)
        if tuple(target_names) == ('valence',):
            table = table.drop(columns=[c for c in table if c.endswith('_arousal')])
        elif tuple(target_names) != ('valence', 'arousal'):
            raise ValueError('Unsupported target names')
        if truth is not None and not truth.equals(exact):
            raise ValueError('Block population targets differ')
        truth = exact
        parts.append(table)
    return pd.concat(parts, ignore_index=True), truth


def block_primary_draws(table, *, iterations, seed, target_names):
    """Production-equivalent shared crossed weights and mean block/grid CCC.

    Return paired cell and primary draws; never drop a block to repair a draw.
    Only the two matched predictors are computed to avoid unrelated statistics.
    """
    keys = ['seed', 'trial_uid']
    targets = ['target_'+x for x in target_names]
    cols = keys+['subject_uid', 'stimulus_uid']+targets
    arrays, ref, grid = [], None, []
    if table.representation.nunique() != 1:
        raise ValueError('Exactly one model/scenario is required')
    for (p, s), cell in table.groupby(['participant_dose', 'stimulus_dose'], sort=True):
        cell = cell.sort_values(keys).reset_index(drop=True)
        if cell.duplicated(keys).any():
            raise ValueError('Duplicate trial in cell')
        if ref is None:
            ref = cell[cols]
        elif not ref.equals(cell[cols]):
            raise ValueError('Different test support or truth')
        grid.append((int(p), int(s)))
        for name in ('prior_personalized', 'combined_personalized'):
            arrays.append(cell[[name+'_'+x for x in target_names]].to_numpy(float))
    if len(grid) != table.participant_dose.nunique()*table.stimulus_dose.nunique():
        raise ValueError('Incomplete rectangular grid')
    _, pi = np.unique(ref.subject_uid.astype(str), return_inverse=True)
    _, si = np.unique(ref.stimulus_uid.astype(str), return_inverse=True)
    n, m = pi.max()+1, si.max()+1
    rng = np.random.default_rng(seed)
    pw = rng.multinomial(n, np.full(n, 1/n), size=iterations)
    sw = rng.multinomial(m, np.full(m, 1/m), size=iterations)
    weights = np.concatenate([np.ones((1, len(ref))), pw[:, pi]*sw[:, si]])
    estimates = np.stack(arrays, axis=1)
    scores, cluster_sizes = [], []
    for block in sorted(ref.seed.unique()):
        mask = ref.seed.eq(block).to_numpy()
        ccc, _ = weighted_scores(ref.loc[mask, targets].to_numpy(float),
                                 estimates[mask], weights[:, mask])
        scores.append(ccc)
        cluster_sizes.extend([ref.loc[mask, 'subject_uid'].nunique(),
                              ref.loc[mask, 'stimulus_uid'].nunique()])
    average = np.mean(scores, axis=0)
    delta = average[:, 1::2]-average[:, ::2]
    valid = np.isfinite(delta[1:]).all(1)
    return delta[0], delta[1:][valid], grid, {
        'valid_bootstrap': int(valid.sum()), 'invalid_bootstrap': int((~valid).sum()),
        'cluster_df': int(min(cluster_sizes)-1), 'n_blocks': len(scores),
        'n_trials': len(ref), 'n_participants': int(n), 'n_stimuli': int(m)}
