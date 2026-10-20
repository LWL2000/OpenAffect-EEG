# V14 manuscript and release integration

Date: 2026-10-20. This work integrates already computed evidence. It does not
rerun a model, bootstrap, simulation, download or server job.

## Evidence coverage

| Evidence | Main manuscript | Supplement and release |
|---|---|---|
| FACED observed EEGNet / adapted LaBraM | Abstract, contributions, primary table and complete matched-increment heatmap | Exact original summary/cell files, full source hashes, protocol and CLI map |
| Same-data five-seed adapted LaBraM | Development-sensitivity rows, distinguished from FACED | Both original analysis sets, one-rotation boundary and source hashes |
| Both full-grid label permutations | Main controls paragraph | Complete intervals and warning decisions for both architectures |
| Two failed sinusoidal controls | Explicit main-text failure statement | Both failed rows, amendments and retained chronology |
| Signed-power diagnostic subsets | Labelled post-failure exploratory evidence | Different realized support (85 participants/23 stimuli), all cells, stopped 52-fit attempt disclosed |
| Equivalence | Pre-outcome margins and average-estimand interpretation | 90% decision intervals, 95% estimation intervals, inconclusive EEGNet, no clinical or cellwise claim |
| Literature audit | Exploratory observation and scope limits | Both original locks, disagreements and post-lock adjudication; no human or representative-prevalence claim |
| Software acceptance | Aggregate reconstruction separated from scientific rerunning | Deterministic v14 rebuild added to the existing quickstart |

`paper/generated/evidence_v14/archive_manifest.json` links 20 immutable source
files across ten completed neural analysis sets. `all_results.csv` retains every
completed neural stage. All 2,500 completed Ridge cell metrics are also archived;
their descriptive mean is -0.0020 CCC, with no new joint interval or equivalence
decision. Original invocation fields are preserved; the paper explicitly
limits their interpretation after the locked-control failures. Old development
figures and interval simulations are retained in development appendices and do
not validate FACED's newer bootstrap.

## Contribution and case strengthening

The nearest participant-only no-EEG baseline is credited. The additional tested
object is a two-axis label-resource surface with fixed support, constant-size
population training and paired predictor contrasts. Arithmetic score
subtraction, identity holdout and no-EEG controls are not claimed individually
new. No exhaustive priority claim is made.

Three existing cases are now explicit: total system versus matched EEG
attribution in two development tasks; the FACED stimulus-resource surface;
and a narrower claim supported by the same pinned EEGain execution trace.
Point gains are not rewritten as significant improvements, descriptive corner
changes are not causal effects, and external classifier scores are not
invalidated. The new case table is regenerated from the existing result CSV.

## Validation and remaining limits

The build audits the main text, v14 result module, v14 supplement, cross-domain
module and all retained development appendices. V14 verification rejects
changed cell files and incorrect equivalence decisions even if a summary's
outer digest is updated. A clean package acceptance also rebuilds the v14
tables and checks the complete release manifest. Plot QA includes color,
grayscale and a whole-page render at the manuscript's actual scale.

Scientific limitations remain explicit: failed locked controls, restricted
architecture/tuning coverage, average rather than cell/target equivalence,
benchmark rather than clinically validated margins, correlated automated
coders, and no new cross-domain empirical study. These cannot be removed by
wording. The contribution is evaluation specificity within these boundaries.

The controlled policy is `configs/submission_release_v14.yaml`, mirrored by
`configs/public_release.yaml`. It contains portable reproduction code, frozen
configurations, protocol amendments, aggregate statistics and this revision.
Host-specific launchers, raw data, trial predictions, weights, private derived
files, and outdated Chinese reading PDFs are outside the current archive. The
original clean model-operated acceptance remains author-assisted. The anonymous
entry must be checked against this exact revision before submission, and the
future venue template must be applied when official instructions are available.
