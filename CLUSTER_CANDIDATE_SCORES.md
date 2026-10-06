Run from this directory (requires NumPy and SciPy):

For direct candidate-pair scoring, run:

```powershell
python candidate_pairwise_scores.py 00026_frenchapproval --output matrices/00026_candidate_pairwise_scores.csv
```

`candidate_pairwise_scores.py` implements B(c,E) over unordered pairs of **distinct candidate approval sets**, without clustering. For each pair S,T it forms U=S\T and V=T\S and adds `min(|V| * |A(c) intersect U|, |U| * |A(c) intersect V|) / n^2`. Nested pairs contribute zero. Duplicate approval columns appear only once in the comparison set, while every original candidate receives a score. Normalization uses the full voter count squared, with no division by the number of comparisons. Identical candidates receive identical scores; replicating every voter equally leaves scores unchanged.

Input can be a headered binary CSV, one `.cat` file, or a directory of `.cat` files. Output is ranked descending, with ties in original candidate order and zero-based candidate IDs. The importable function is `candidate_pairwise_scores(matrix)`. Run its mathematical checks with `python -m unittest test_candidate_pairwise_scores`.

For clustering-based scoring, run:

```powershell
python cluster_candidate_scores.py 00026_frenchapproval/00026-00000001.cat --clustering complete --cutoff 0.5 --scoring pairwise
python cluster_candidate_scores.py matrices/frenchapproval.csv --clustering average --cutoff 0.4 --scoring nash --output matrices/scores.csv --groups-output matrices/groups.csv
python cluster_candidate_scores.py 00026_frenchapproval --clustering correlation --scoring egalitarian
```

Directory inputs combine all `.cat` elections, requiring identical candidate names and order. Candidate Jaccard distances are computed over the entire input electorate. CSV input has candidate names in the first row and one binary approval ballot per subsequent row.

For voters x,y, the script takes the maximum candidate Jaccard distance over the candidates on which they disagree. Empty and singleton disagreement sets yield zero. This dissimilarity need not satisfy the triangle inequality; distinct ballots can have distance zero.

Both scoring and plotting accept `--distance disagreement|hamming|jaccard`:

- `disagreement` (default): the original maximum candidate Jaccard distance described above.
- `hamming`: the fraction of candidate columns where two voters differ, including never-approved candidates in the denominator. This is normalized Hamming, not the raw count used by some other scripts in this repository.
- `jaccard`: Jaccard distance between the voters' approved-candidate sets directly; two empty ballots have distance zero.

All three distances are in [0,1]. Cutoffs and the correlation objective use the selected distance; candidate scoring formulas are unchanged. In Python, select with `voter_distances(matrix, distance="jaccard")`.

```powershell
python cluster_candidate_scores.py 00026_frenchapproval --distance jaccard --clustering complete --cutoff 0.99
python plot_jaccard_group_blobs.py --distance hamming --cutoffs 0.25 0.5 0.75
```

`--clustering complete` uses maximum intergroup dissimilarity; `average` uses the mean over all cross-group voter pairs. Both merge at heights less than or equal to `--cutoff` (default 0.5). `correlation` starts with singleton voters and repeatedly chooses the merge that most reduces the stated correlation objective, stopping when no merge improves it. It is a deterministic greedy heuristic, with no global optimality guarantee; the cutoff is unused. None of these methods requires k. Ties can depend on input row order.

With group approval fractions p_i and sizes n_i, scoring choices are:

- `pairwise` (default): sum over i<j of n_i n_j min(p_i,p_j), without normalization. One group gives zero for every candidate.
- `nash`: product of p_i raised to n_i/n, evaluated in log space. This population-weighted geometric mean preserves the ranking of the product with exponents n_i. Any zero group approval gives zero; no smoothing is applied.
- `egalitarian`: min_i p_i, the worst per-voter group approval fraction. Population is handled by dividing each group's approval count by its size; no additional size weight is applied to the minimum.

The script prints scores in descending order and the correlation objective of the selected partition. Optional output CSVs contain candidate IDs/names/scores and voter IDs/group IDs; IDs are zero-based and voter IDs follow input order. Functions `cluster_voters(distances, algorithm, cutoff)` and `score_candidates(matrix, labels, scoring)` are also importable.

Distances and clustering use O(n²) memory for n voters. The greedy correlation method performs O(n³) work in the worst case and is intended for modest datasets. Duplicate ballots share distance calculations but are retained as individual voters during clustering. The full combined dataset may require substantial memory.

Check the mathematical cases with `python -m unittest test_cluster_candidate_scores`.

To plot groups using the full combined `00026_frenchapproval` electorate (2,597 voters), with partition distance D for both MDS and clustering:

```powershell
python plot_jaccard_group_blobs.py
```

This writes one PNG and a voter/group membership CSV for each cutoff (0.9, 0.95, 0.975, 0.99) to `preference_map_outputs/disagreement_mds/group_blobs/`. Full-election filenames include `_full_` to distinguish them from the earlier sample plots. The CSVs contain the newly computed x/y coordinates. It uses complete linkage by default; `--clustering average` selects average linkage. Use `--input` to choose a different .cat file, directory, or headered approval CSV. `--output-dir` and `--cutoffs` can override the defaults.

The plotting and scoring scripts now load the full election with the same loader and voter order. Candidate approver-set distances and clustering therefore use the same electorate. The exact same candidate-disagreement distance matrix D is passed to metric MDS and hierarchical clustering. MDS is fitted once per run and its normalized stress is shown in each title. D need not be a metric, so the two-dimensional embedding is approximate. All images keep the same coordinates and axis limits. Blobs use the approver plot's disk-union method. Coincident (or numerically indistinguishable) points assigned to different groups are marked with black crosses and omitted from enclosing blobs, because the fixed embedding cannot separate them. Group IDs are the smallest zero-based input row in each group; legends list up to the 24 largest groups, while every group is plotted and saved in the CSV. These group IDs can differ from the scoring script's consecutive labels, but group membership matches.

The default `--mds-distance clustering` follows `--distance`, so clustering and MDS use the same matrix. You can explicitly choose `--mds-distance disagreement`, `hamming`, or `jaccard` to embed with a different distance. Output folders and filenames identify the embedding distance; filenames also identify the clustering distance when it differs. The default disagreement outputs keep their existing names.

To reproduce the earlier 1,500-voter sample plots, explicitly select the saved coordinate file:

```powershell
python plot_jaccard_group_blobs.py --coordinates preference_map_outputs/jaccard_mds/00026_frenchapproval_jaccard_mds.csv
```

Add `--mds-distance saved-jaccard` to that command to use the original ballot-Jaccard layout. Sample-only clustering produces 4 groups at cutoff 0.99; full-election clustering produces 9.

For normalized Hamming on a saved sample, also pass `--candidate-count 16` for dataset 00026 (or the correct full slate size for another dataset). Saved approved-candidate lists alone cannot reveal candidates who received no sampled approvals. Full-election inputs already contain all candidate columns and need no extra argument. The group counts above refer to the default disagreement distance.
