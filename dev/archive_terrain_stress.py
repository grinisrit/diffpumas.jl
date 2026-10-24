#!/usr/bin/env python3
"""Adopt the complete original terrain queue into a fresh canonical review bundle.
No numerical production, manuscript installation, or generated conclusion prose.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tomllib

import summarize_terrain_stress as summary

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def dump_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve()
    if output.exists():raise ValueError('fresh canonical output required')
    plan=json.loads((root/'preregistered_plan.json').read_text());complete=json.loads((root/'queue_complete.json').read_text())
    if complete.get('execution_completed') is not True or complete.get('cases')!=[case['name'] for case in plan['cases']]:
        raise ValueError('require complete original sequential queue, no substituted orchestration')
    for name,expected in plan['source_and_input_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if not path.is_file() or digest(path)!=expected:raise ValueError('computed source/input no longer matches: '+str(path))
    ledger=[];refits=[];diagnostics=[];payload=summary.summarize(root)
    if not payload['complete'] or len(payload['cases'])!=10:raise ValueError('incomplete adopted outcomes')
    by_name={row['case']:row for row in payload['cases']}
    for case in plan['cases']:
        folder=root/case['name'];proof=summary.verified_case(folder)
        end=json.loads((folder/'end_status.json').read_text());execution=json.loads((folder/'execution.json').read_text())
        for name,expected in end['output_sha256'].items():
            if digest(folder/name)!=expected:raise ValueError('case output changed: '+str(folder/name))
        command=case['command'];expected_inner=command[command.index('--')+1:]
        if execution['command']!=expected_inner or execution['cpu_pool']!=list(range(8)) or execution['address_space_bytes']!=16*1024**3:
            raise ValueError('case deviated from original scientific command/resource allocation: '+case['name'])
        if execution['julia_runtime']!=plan['julia_runtime']:raise ValueError('runtime identity differs')
        recorded={str((Path(name) if Path(name).is_absolute() else ROOT/name).resolve()):value for name,value in execution['source_sha256'].items()}
        for name,expected in plan['source_and_input_sha256'].items():
            path=(Path(name) if Path(name).is_absolute() else ROOT/name).resolve()
            if recorded.get(str(path))!=expected:raise ValueError('case lacks planned source/input binding: '+str(path))
        report=tomllib.loads((folder/'data/report.toml').read_text());row=by_name[case['name']]
        if report['grid_cells']!=600 or report['support_cells']!=147 or report['covered_bins']!=1003 or report['absolute_prior_weight']!=3.954930197858131:
            raise ValueError('fixed grid/support/objective context changed')
        if not all(report.get(key) is True for key in ('baseline_prediction_bitwise','zero_geometry_bitwise','new_path_cache_primal_bitwise')):
            raise ValueError('case parity gate missing')
        ledger.append(dict(case=case['name'],end_status_sha256=digest(folder/'end_status.json'),execution_sha256=digest(folder/'execution.json'),
            output_sha256=end['output_sha256'],actual_cpu_pool=execution['cpu_pool'],runtime=execution['julia_runtime'],all_original_source_bindings_verified=True))
        diagnostics.append(dict(case=case['name'],screen=case['screen'],assumed_peak_height_fraction=case['amplitude'],
            forward_only=case['forward_only'],fixed_original_field_sector_rms_log_residual=row['sector_fixed_field']['rms_log_residual'],
            fixed_original_field_sector_mean_log_residual=row['sector_fixed_field']['mean_log_residual'],
            fixed_original_field_max_relative_flux_change=row['fixed_field_flux_max_relative_to_zero'],
            max_realized_map_height_change_m=report['max_abs_map_height_change_m'],max_realized_slant_change_m=report['max_abs_slant_change_m']))
        if not case['forward_only']:
            refits.append(dict(case=case['name'],screen=case['screen'],assumed_peak_height_fraction=case['amplitude'],
                field_relative_to_matching_zero=row['field_relative_to_matching_zero'],field_relative_to_old_screened_baseline=report['field_relative_change_from_old_baseline'],
                sector_refit_rms_log_residual=row['sector_refit']['rms_log_residual'],sector_rms_change_from_matching_zero=row['sector_refit']['rms_change_from_matching_zero'],
                objective=report['objective'],projected_gradient_inf=report['projected_gradient_inf'],stationary_at_requested_tolerance=report['stationary_at_requested_tolerance'],
                recorded_iterations=report['iterations_recorded'],fit_bins=report['fit_bins'],sector_fit_bins=report['sector_fit_bins'],
                max_realized_map_height_change_m=report['max_abs_map_height_change_m'],max_realized_slant_change_m=report['max_abs_slant_change_m'],
                off_sector_changed_paths=report['off_sector_changed_paths'],off_sector_max_slant_change_m=report['off_sector_max_abs_slant_change_m'],
                changed_supported_cell_intersections=report['changed_supported_cell_intersections']))
    output.mkdir(parents=True)
    for name in ('preregistered_plan.json','queue_complete.json'):shutil.copy2(root/name,output/name)
    (output/'terrain_summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    dump_csv(output/'terrain_refit_comparison.csv',refits);dump_csv(output/'terrain_fixed_field_diagnostics.csv',diagnostics)
    summary.figures(payload,output)
    import matplotlib
    provenance=dict(matplotlib_version=matplotlib.__version__,execution_completed=True,original_controller_completed=True,all_ten_original_scientific_commands_exact=True,
        no_parallel_numerical_cases_or_output_relocations=True,helper_sha256=digest(Path(__file__)),summary_helper_sha256=digest(Path(summary.__file__)),
        python=platform.python_version(),cases=ledger,
        scope='Assumed terrain stress and finite-budget solver outcomes; no measured DEM errors, attribution of residual cause, converged sensitivity or uncertainty bounds.',
        forward_only_scope='The four ±1% rows use fixed_field columns only. Generic refit_log_residual source columns equal the unchanged original field there and are not refit responses.',
        outputs_sha256={path.name:digest(path) for path in output.iterdir() if path.is_file()})
    (output/'canonical_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(output)
if __name__=='__main__':main()
