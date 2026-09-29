# Jaccard analysis options

Use `--metric jaccard` with the local Hamming-based analysis scripts:

- `generic_dc_mpjr_min_gamma.py`, `dc_mpjr_verify.py`
- `batch_proportional_audit.py`, `proportional_audit.py`, `candidate_pf_pareto.py`
- `representation_level_audit.py`, `wasserstein_approver_distance.py`
- `clear_communities.py`
- `plot_voter_dendrogram.py`, `compute_candidate_group_cover.py`
- `nested_greedy_capture.py`, `experiment_pf_by_level.py`

`approval_dap_candidate_effects.py` already supports Jaccard: use
`--metrics jaccard`. Its existing default calculates both metrics.

Defaults remain unchanged. The batch audit historically uses Euclidean for PF
and Hamming for minimum gamma; specifying `--metric` applies that metric to both.
The PF experiment inherits the metric from the group JSON. An explicit metric
must match that JSON; regenerate the groups before changing the experiment metric.

Jaccard is `|A symmetric-difference B| / |A union B|` for binary approval sets.
Two empty ballots have distance zero; one empty and one nonempty ballot have
distance one. Nonbinary Jaccard input is rejected. Raw Hamming remains the
number of differing coordinates, without normalization.

Clustering accepts fractional `--max-height`. Defaults are 4 for Hamming and
0.5 for Jaccard. These thresholds are separate choices, not equivalent cuts.
Jaccard height summaries include every distinct merge height and the selected cut.
Gamma witnesses and clear-community diameters retain fractional distances.
Minimum gamma is reconstructed as an exact rational using the bounded union sizes.

## Reproduce the reruns

```powershell
python run_jaccard_analyses.py --section analyses
python run_jaccard_analyses.py --section hierarchy
python run_jaccard_analyses.py --section dap
```

Outputs are in `matrices/jaccard/`; previous results are preserved. The default
input is the current `matrices/frenchapproval.csv` (20,076 voters, 11 candidates).
Older saved Hamming hierarchy/clustering outputs used a different, 2,597-voter
election and are not direct comparisons to these reruns. Use `--csv` and
`--output-dir` on the runner to choose a different matrix and destination.

The analysis section runs candidate audits for all 11 candidates. Its generic
minimum-gamma CLI example uses the first candidate's distinct potential-approver
locations as M and distinct actual-approver locations as X; those input CSVs are
saved with the outputs. The batch and candidate-specific DC audits instead retain
one center per voter, including multiplicities, as before.

The hierarchy section builds all 20,076 levels and uses 20 trials per distinct
group family with seed 0. Duplicate ballot compression preserves voter
multiplicities and original row-order tie breaking. The DAP section uses that
script's configured French datasets, including the pooled 00026 election.

Each section writes a manifest with exact commands, exit codes, timings, and
the matrix hash; each command has a separate log. Generated results are ignored
by Git under the existing `matrices/` rule.
