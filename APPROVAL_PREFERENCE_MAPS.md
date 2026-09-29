# Approval-to-ordinal preference maps

This workflow converts the two French PrefLib approval datasets into sampled
strict rankings, then maps those rankings beside two distributions used in the
IJCAI23 code: impartial culture and normalized Mallows (`norm-phi = 0.2`).

## Run the complete workflow

From this directory, with NumPy and Matplotlib installed:

```powershell
python run_french_preference_maps.py
```

The defaults draw 250 ordinal rankings from every `.cat` election and use at
most 75 rankings per source in each map. Outputs are placed under
`preference_map_outputs/`:

- `ordinal_samples/00026_frenchapproval/*.soc`
- `ordinal_samples/00073_frenchapproval/*.soc`
- `00026_frenchapproval_preference_map.png` and its coordinate CSV
- `00073_frenchapproval_preference_map.png` and its coordinate CSV

The datasets are mapped separately because dataset 00026 has 16 candidates and
dataset 00073 has 11. Kendall distances between rankings on different candidate
sets are not comparable.

For a quicker smoke run:

```powershell
python run_french_preference_maps.py --num-rankings 40 --map-per-source 20
```

## Sampling semantics

For each requested ranking, `sample_approval_rankings.py`:

1. draws an approval ballot in proportion to its multiplicity in the election;
2. uniformly permutes its approved candidates;
3. uniformly permutes its disapproved candidates; and
4. puts the approved tier above the disapproved tier.

For a ballot with `k` approved candidates out of `m`, this samples each of its
`k! (m-k)!` consistent strict rankings with equal probability. Empty and full
approval ballots are handled naturally. The seed makes the result reproducible.

Use the sampler independently like this:

```powershell
python sample_approval_rankings.py 00073_frenchapproval `
  --num-rankings 500 --seed 23 --output-dir preference_map_outputs/my_samples
```

The default `resample` mode produces any requested sample size. To retain every
original voter exactly once and only randomize within approval tiers, use
`--mode complete` and omit `--num-rankings`.

The `.soc` output is aggregated and uses zero-based candidate IDs, as expected
by the Mapel version bundled with the IJCAI23 repository.

## Mapping semantics

`map_ordinal_preferences.py` reads one or more sampled `.soc` files sharing the
same candidate slate. It adds the selected synthetic cultures, computes
normalized Kendall/swap distances, applies classical multidimensional scaling,
and writes both a PNG and the underlying coordinates.

```powershell
python map_ordinal_preferences.py preference_map_outputs/my_samples/*.soc `
  --output preference_map_outputs/my_map.png `
  --cultures impartial mallows --mallows-norm-phi 0.2
```

The Mallows generator uses the repeated-insertion model and converts Mapel's
normalized dispersion to ordinary Mallows `phi` by matching expected inversion
distance. Its central ranking is the input candidate order.

## Placing the French elections on the IJCAI23 DAP map

The DAP map compares whole elections rather than individual rankings. Its 302
stored elections all contain exactly 8 candidates and 96 voters. Run:

```powershell
python map_french_on_dap.py
```

The script performs the following compatibility transformation:

1. pools all voters from the six `00026` districts into one approval election;
2. selects the eight candidates with the largest pooled approval totals;
3. does the same eight-candidate selection for `00073`;
4. samples 96 approval-consistent rankings from each projected election; and
5. places the elections on the existing `diversity_map_8_96` coordinates.

Candidate selection is recorded rather than silently discarded. See
`preference_map_outputs/dap/selected_candidates.csv` for original IDs, names,
approval totals, and projected IDs.

The original 302 map points and their published MDS coordinates remain fixed.
For each French election, the script computes exact anonymous swap distances
to 20 well-spread DAP landmark elections using the C++ implementation bundled
with the IJCAI23 code. A landmark-distance k-nearest-neighbor model places the
new point out of sample. The value of `k` is selected by leave-one-out error on
the original DAP elections. Thus the overlay does not rerun or perturb the
paper's MDS embedding. Placement uncertainty, cross-validation error, and the
nearest original elections are recorded in
`preference_map_outputs/dap/french_dap_placements.csv`.

Outputs are:

- `preference_map_outputs/dap/french_on_dap_map.png`;
- two sampled `8x96.soc` elections;
- `selected_candidates.csv`;
- `french_dap_placements.csv`; and
- cached exact distances in `landmark_distances.csv`.

The bundled binary is a CPython 3.10 Windows extension. Exact distances are
cached, so rerunning with the same seed and landmarks is fast.

## Approval-ballot maps with Jaccard MDS

To map the approval ballots directly, without generating ordinal completions:

```powershell
python plot_french_jaccard_mds.py
```

Each point is a sampled voter. For approval sets `A` and `B`, MDS receives the
Jaccard dissimilarity

```text
1 - |A intersection B| / |A union B|.
```

This is `1 - Jaccard similarity`, because MDS requires a dissimilarity matrix.
Two empty ballots are assigned distance zero. Sampling is without replacement
and respects the original PrefLib ballot multiplicities.

The six elections in `00026` share a single MDS embedding and are colored by
district. Dataset `00073` is embedded separately because its candidates form a
different set. Metric SMACOF MDS is initialized from a classical MDS solution,
and normalized stress is printed on each plot.

The defaults use at most 250 voters from each election. Change that with:

```powershell
python plot_french_jaccard_mds.py --max-voters-per-election 500
```

Images and coordinates are written to `preference_map_outputs/jaccard_mds/`.

## Approval diversity, agreement, and polarization

[`approval_dap_candidate_effects.py`](approval_dap_candidate_effects.py) adapts
Definitions 2--4 and the agreement/1-Kemeny relation from the IJCAI23 paper to
approval ballots. Run:

```powershell
python approval_dap_candidate_effects.py
```

For a ballot metric `d`, let `kappa_k` be the total distance from all voters to
their closest one of `k` representative ballots. The reported indices are:

```text
agreement    = 1 - 2*kappa_1 / (n*diameter)
diversity    = sum_k (kappa_k/k) / (n*diameter)
polarization = 2*(kappa_1-kappa_2) / (n*diameter)
```

Raw Hamming distance has `diameter = number of candidates`; Jaccard distance
has `diameter = 1`. As in the scalable algorithm shipped with the paper,
representatives are selected greedily from observed ballot types. Ballot types
remain multiplicity-weighted, so this uses the complete electorate rather than
a voter sample.

The default analyzes the pooled six-district `00026` profile and the `00073`
profile. For every candidate and both metrics, it removes that candidate from
all ballots, merges newly identical ballot types, recalculates all three
indices, and reports:

```text
delta = score after removal - original score.
```

Results are written to
`preference_map_outputs/approval_dap_candidate_effects.csv`. Baseline rows have
an empty `removed_candidate` field. To additionally analyze each `00026`
district separately, use `--include-individual-00026`.

The factor `2` in the paper's agreement normalization is tailored to swap
distance. With Jaccard distance, a 1-median cost can exceed half of
`n*diameter`, so a direct substitution can produce negative agreement values.
The script deliberately reports these values without clipping or silently
changing the paper's formula; the CSV includes a normalization note on every
row.

