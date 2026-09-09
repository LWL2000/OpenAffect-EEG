"""Render all frozen matched contrasts and summarize existing learning evidence."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openaffect_eeg.artifacts import sha256_file


def coverage_assets(root, output):
    source = root/'results/block_validation_v12/summary.csv'
    summary = pd.read_csv(source)
    summary.to_csv(output/'five_block_all_coverage.csv', index=False)
    uniform = summary.loc[summary.estimand.eq('uniform_grid_mean')]
    if len(uniform) != 24 or not uniform.replicates.eq(300).all():
        raise ValueError('Incomplete five-block validation')
    uniform.to_csv(output/'five_block_uniform_coverage.csv', index=False)
    lines = [r'\begin{table}[ht]\centering\small',
        r'\caption{Five-block uniform-mean coverage, 300 replicates per row and a nominal 95\% interval. Brackets are Wilson Monte Carlo intervals for percentile coverage, not EEG effect intervals.}',
        r'\begin{tabular}{llccc}\toprule',
        r'Task & Scenario & Percentile [MC interval] & Basic & t-normal\\\midrule']
    for dataset, task in [('ds005540', 'Emo'), ('ds006850', 'Urban')]:
        for scenario, label in [('trial_signal', 'Trial signal'),
                ('participant_constant', 'Participant constant'),
                ('independent_noise', 'Independent noise'), ('labels_only', 'Labels only')]:
            group = uniform.loc[uniform.dataset_id.eq(dataset)
                & uniform.scenario.eq(scenario)].set_index('method')
            row = group.loc['percentile']
            lines.append(f'{task} & {label} & {row.coverage:.3f} '
                f'[{row.coverage_mc_low:.3f}, {row.coverage_mc_high:.3f}] & '
                f'{group.loc["basic"].coverage:.3f} & {group.loc["normal_t"].coverage:.3f}'+r'\\')
    lines += [r'\bottomrule\end{tabular}', r'\end{table}']
    (output/'five_block_coverage.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'paper/generated/final_closure_v9/percentile/table_added_value_contrasts.csv'
    output = root/'paper/generated/reviewer_revision_v12'
    output.mkdir(exist_ok=True)
    coverage_assets(root, output)
    table = pd.read_csv(source)
    selected = table.loc[table.contrast.eq('eeg_added_matched_calibration')].copy()
    keys = ['dataset_id', 'representation', 'participant_dose', 'stimulus_dose']
    if selected.duplicated(keys).any() or len(selected) != 300:
        raise ValueError('Expected all 300 unique task-setting-dose contrasts')
    selected.to_csv(output/'matched_increment_all_cells.csv', index=False)
    corners = selected.loc[selected.participant_dose.isin([0, 8]) & selected.stimulus_dose.isin([0, 8])]
    corners.to_csv(output/'deployment_corners.csv', index=False)
    limit = float(np.ceil(selected.ccc_delta.abs().max()*100)/100)
    doses = [0, 1, 2, 4, 8]
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':7,
                         'pdf.fonttype':42, 'svg.fonttype':'none'}):
        fig, axes = plt.subplots(4, 3, figsize=(6.5, 6.7), layout='constrained')
        for ri, (dataset, suffix, eegsuffix, row_name) in enumerate([
            ('ds005540','ridge_base','lr1e3','EmoEEG-MC / setting 1'),
            ('ds005540','ridge_strong','lr1e4','EmoEEG-MC / setting 2'),
            ('ds006850','ridge_base','lr1e3','Urban / setting 1'),
            ('ds006850','ridge_strong','lr1e4','Urban / setting 2')]):
            models = [('de' if dataset=='ds005540' else 'bandpower')+'_'+suffix,
                      'labram_'+suffix, 'eegnet_'+eegsuffix]
            for ci, model in enumerate(models):
                ax = axes[ri,ci]
                cell = selected.loc[selected.dataset_id.eq(dataset)&selected.representation.eq(model)]
                z = cell.pivot(index='participant_dose',columns='stimulus_dose',values='ccc_delta').reindex(index=doses,columns=doses).to_numpy()
                if z.shape != (5,5) or not np.isfinite(z).all():
                    raise ValueError('Missing/nonfinite plot cell: '+model)
                im = ax.imshow(z*1000, origin='lower', vmin=-limit*1000, vmax=limit*1000, cmap='RdBu',
                               interpolation='nearest', aspect='equal')
                ax.set_xticks(range(5), doses)
                ax.set_yticks(range(5), doses)
                ax.tick_params(length=0, labelsize=6)
                if ri==0:ax.set_title(['Band-power Ridge','Frozen LaBraM','Standard EEGNet'][ci],fontsize=8)
                if ci==0:ax.set_ylabel(row_name+'\nCalibration trials',fontsize=7)
                if ri==3:ax.set_xlabel('Ratings per test stimulus',fontsize=7)
                for y in range(5):
                    for x in range(5):
                        label = f'{z[y,x]*1000:+.0f}'
                        if abs(z[y,x])<.0005:label='0'
                        ax.text(x,y,label,ha='center',va='center',fontsize=5.7,
                                color='white' if abs(z[y,x])>.65*limit else 'black')
                for spine in ax.spines.values():spine.set_visible(False)
        cb=fig.colorbar(im,ax=axes,orientation='horizontal',fraction=.025,pad=.025,shrink=.65)
        cb.set_label('Matched EEG increment: [CCC(EC) − CCC(PC)] × 1000',fontsize=8)
        fig.savefig(output/'matched_increment_heatmap.pdf')
        fig.savefig(output/'matched_increment_heatmap.png',dpi=250)
        plt.close(fig)
    verification_path=root/'paper/generated/final_closure_v9/verification/verification.json'
    verification=json.loads(verification_path.read_text())
    learning=[]
    for dataset, record in verification['datasets'].items():
        data=pd.DataFrame(record['learning'])
        for branch, group in data.groupby('branch'):
            learning.append(dict(dataset_id=dataset,branch=branch,fit_records=len(group),
                selected_epoch_one=int(group.selected_epochs.eq(1).sum()),
                selected_epoch_median=float(group.selected_epochs.median()),
                selected_epoch_min=int(group.selected_epochs.min()),
                selected_epoch_max=int(group.selected_epochs.max()),
                median_objective_reduction=float((group.initial_train_objective-group.final_train_objective).median())))
    pd.DataFrame(learning).to_csv(output/'learning_diagnostics.csv',index=False)
    findings=[]
    for (dataset,model), group in selected.groupby(['dataset_id','representation']):
        findings.append(dict(dataset_id=dataset,representation=model,
            minimum=float(group.ccc_delta.min()), maximum=float(group.ccc_delta.max()),
            positive_pointwise=int(group.ccc_ci_low.gt(0).sum()),
            negative_pointwise=int(group.ccc_ci_high.lt(0).sum()),
            positive_dose_family=int(group.ccc_simultaneous_low.gt(0).sum()),
            negative_dose_family=int(group.ccc_simultaneous_high.lt(0).sum())))
    (output/'heterogeneity_summary.json').write_text(json.dumps(findings,indent=2)+'\n')
    (output/'figure_manifest.json').write_text(json.dumps({
        'source_sha256':sha256_file(source), 'learning_source_sha256':sha256_file(verification_path),
        'script_sha256':sha256_file(Path(__file__)), 'cells':300, 'corner_rows':len(corners),
        'color_limits':[-limit,limit], 'dimensions_inches':[6.5,6.7],
        'estimand':'Mean of five block-level paired CCC differences at each dose.',
        'uncertainty':'Full conditional pointwise and within-model dose-family intervals in companion CSV.',
        'rounding':'Heatmap values and color scale multiplied by 1000; rounded integers are not exact zeros.',
        'selection':'All 12 grids. Post-analysis presentation; no new fitting or hypothesis test.'},indent=2)+'\n')
    print(json.dumps({'cells':len(selected),'corners':len(corners),'learning':learning},indent=2))


if __name__=='__main__':
    main()
