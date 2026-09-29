"""Rerun the approval analyses with Jaccard distance, preserving old outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from generic_dc_mpjr_min_gamma import load_matrix
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'matrices/00026_frenchapproval.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'matrices/jaccard')
    parser.add_argument('--section', choices=['analyses', 'hierarchy', 'dap'], default='analyses')
    parser.add_argument('--resume', action='store_true', help='Skip successful commands in this section manifest')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    names, voters = load_matrix(args.csv)
    records = []
    manifest_path = out / ('manifest_' + args.section + '.json')
    input_hash = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['sha256'] != input_hash:
            raise ValueError('Cannot resume: input matrix changed')
        records = previous['runs']
    def run(script, *options):
        command = [sys.executable, str(ROOT / (script + '.py')), *map(str, options)]
        if args.resume and any(r['command'] == command and r['exit_code'] == 0 for r in records):
            return
        label = script + ('_' + str(len(records)))
        print('Running', script, *map(str, options), flush=True)
        started = time.time()
        with (out / (label + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONIOENCODING': 'utf-8'})
        records.append(dict(command=command, exit_code=result.returncode,
                            seconds=round(time.time()-started, 3), log=label+'.log'))
        manifest = dict(input=str(args.csv.resolve()), sha256=input_hash,
                        population_size=len(voters), candidates=names, metric='jaccard',
                        clustering_cut_height=.5, trials=20, seed=0, runs=records)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{script} failed: see {out / (label + ".log")}')
        print('Finished', script, f'{time.time()-started:.1f}s', flush=True)
    if args.section == 'dap':
        run('approval_dap_candidate_effects', '--metrics', 'jaccard', '--output', out / 'approval_dap_candidate_effects.csv')
    elif args.section == 'hierarchy':
        run('nested_greedy_capture', args.csv, '--metric', 'jaccard', '--output', out / 'nested_greedy_capture_groups.json')
        run('experiment_pf_by_level', '--voters', args.csv, '--groups', out / 'nested_greedy_capture_groups.json',
            '--metric', 'jaccard', '--output', out / 'pf_by_greedy_capture_level.csv')
    else:
        run('clear_communities', args.csv, '--metric', 'jaccard', '--output', out / 'clear_communities.csv',
            '--partition-output', out / 'clear_community_partition.csv')
        run('candidate_pf_pareto', args.csv, '--metric', 'jaccard', '--output', out / 'candidate_pf_by_k.csv',
            '--dominance-output', out / 'candidate_pf_dominance.csv')
        run('wasserstein_approver_distance', args.csv, '--metric', 'jaccard', '--output', out / 'wasserstein_approver_distance.csv')
        run('batch_proportional_audit', args.csv, '--metric', 'jaccard', '--output', out / 'batch_proportional_audit.csv')
        for i, name in enumerate(names):
            prefix = f'candidate_{i+1:02d}'
            run('representation_level_audit', args.csv, '--candidate', name, '--metric', 'jaccard',
                '--output', out / (prefix + '_representation.csv'), '--plot', out / (prefix + '_representation.png'))
            run('dc_mpjr_verify', args.csv, '--candidate', name, '--metric', 'jaccard')
            run('proportional_audit', args.csv, '--candidate', name, '--metric', 'jaccard')
        # Generic CLI requires unique candidate rows: a distinct-location instance.
        chosen = voters[voters[:, 0] == 1]
        potential = voters.copy(); potential[:, 0] = 1
        for filename, data in [('generic_candidates.csv', potential), ('generic_selected.csv', chosen)]:
            with (out / filename).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle); writer.writerow(names)
                writer.writerows(np.unique(data, axis=0).astype(int))
        run('generic_dc_mpjr_min_gamma', args.csv, out / 'generic_candidates.csv', out / 'generic_selected.csv', '--metric', 'jaccard')
        run('plot_voter_dendrogram', '--csv', args.csv, '--metric', 'jaccard', '--max-height', '.5',
            '--out', out / 'voter_dendrogram.png', '--metrics-csv', out / 'voter_group_sizes_by_height.csv')
        run('compute_candidate_group_cover', '--csv', args.csv, '--metric', 'jaccard', '--max-height', '.5',
            '--cover-csv', out / 'candidate_group_cover.csv')


if __name__ == '__main__':
    main()
