#!/usr/bin/env python3
"""Isolated exact-ray-cache fit variant; never rewrites the frozen example.

--prepare-only writes auditable transformed Julia without executing it. Actual
execution requires a reviewed parity certificate. Invoke through run_revision_job
so all inputs, generated text, outputs and the Julia runtime remain end audited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/lvd_tomography.jl'
HELPER = ROOT / 'dev/prototype_stability_guarded_ray_cache.jl'
ORIGINAL_SHA = 'dc4e25dce955c53eddb965f7b97bec2eae78703658e9d5ef530f1f7fddd676c5'
START = '    flux_only_at = (ρ, wv) -> begin\n'
END = '\n    # Data-driven outlier rejection using the centered dry model/measured log ratio.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(folder):
    folder = Path(folder)
    end = json.loads((folder / 'end_status.json').read_text())
    if end.get('returncode') != 0 or not all(end.get(k) is True for k in
            ('execution_completed', 'source_unchanged', 'runtime_unchanged')):
        raise ValueError(f'unclean parity gate: {folder}')
    for name, digest in end['output_sha256'].items():
        if sha(folder / name) != digest:
            raise ValueError(f'gate output changed: {folder/name}')
    return end


def recorded_digest(recorded, path):
    path = Path(path)
    relative = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else None
    return recorded.get(str(path), recorded.get(relative))


def verify_certificate(path):
    cert = json.loads(Path(path).read_text())
    if cert.get('schema_version') != 1 or cert.get('approved_for_successor_queue') is not True:
        raise ValueError('certificate must explicitly approve the successor implementation')
    inputs = cert.get('implementation_sha256', {})
    for p in (EXAMPLE, HELPER, Path(__file__).resolve()):
        if inputs.get(str(p)) != sha(p):
            raise ValueError(f'certificate does not bind actual implementation: {p}')
    transformed, _, _ = transformed_source()
    if cert.get('transformed_source_sha256') != hashlib.sha256(transformed.encode()).hexdigest():
        raise ValueError('certificate does not bind exact transformed Julia text')
    context = cert.get('certified_context_sha256', {})
    required = [ROOT/'Project.toml', ROOT/'Manifest.toml', *sorted((ROOT/'src').glob('*.jl'))]
    for p in required:
        if context.get(str(p)) != sha(p):
            raise ValueError(f'uncertified core environment/source: {p}')
    for p, digest in context.items():
        if not Path(p).is_absolute() or sha(p) != digest:
            raise ValueError(f'certified context changed: {p}')
    gates = cert.get('parity_gates', {})
    for name in ('dry_serial', 'corrected_serial', 'corrected_threaded', 'actual_gn_trial'):
        gate = gates.get(name, {})
        folder = Path(gate.get('path', ''))
        if gate.get('passed') is not True or not folder.is_absolute():
            raise ValueError(f'missing successful parity gate: {name}')
        end = audit(folder)
        execution = json.loads((folder/'execution.json').read_text())
        recorded = execution.get('source_sha256', {})
        if execution.get('julia_runtime') != cert.get('julia_runtime'):
            raise ValueError(f'gate Julia runtime differs: {name}')
        for context_path, digest in context.items():
            if recorded_digest(recorded, context_path) != digest:
                raise ValueError(f'gate lacks certified context input: {name}: {context_path}')
        for p in (EXAMPLE, HELPER, Path(__file__).resolve()):
            if recorded_digest(recorded, p) != sha(p):
                raise ValueError(f'gate used a different implementation: {name}: {p}')
        proof_rel = gate.get('proof_path')
        if not proof_rel or Path(proof_rel).is_absolute() or '..' in Path(proof_rel).parts:
            raise ValueError(f'missing bound semantic proof: {name}')
        if end['output_sha256'].get(proof_rel) != gate.get('proof_sha256'):
            raise ValueError(f'proof is not end bound: {name}')
        proof = json.loads((folder/proof_rel).read_text())
        checks = proof.get('checks', {})
        required_checks = ['full_prediction_bitwise', 'residual_bitwise', 'profile_bitwise',
                           'objective_bitwise', 'context_guard_passed']
        if proof.get('schema_version') != 1 or proof.get('gate_kind') != name:
            raise ValueError(f'wrong semantic proof identity: {name}')
        for key in ('closure_factory_path', 'closure_factory_sha256', 'transformed_source_sha256'):
            if proof.get(key) != cert.get(key):
                raise ValueError(f'proof factory/transformation mismatch: {name}: {key}')
        factory = Path(cert['closure_factory_path'])
        if not factory.is_absolute() or sha(factory) != cert['closure_factory_sha256']:
            raise ValueError('closure factory changed')
        if recorded_digest(recorded, factory) != sha(factory):
            raise ValueError(f'factory is not a recorded gate input: {name}')
        if name == 'actual_gn_trial': required_checks += ['trial_stream_complete']
        if proof.get('passed') is not True or any(checks.get(k) is not True for k in required_checks):
            raise ValueError(f'incomplete semantic parity proof: {name}')
        for key in ('scope','sample_count','case_count','threads','cpu_affinity'):
            if not proof.get(key): raise ValueError(f'proof lacks {key}: {name}')
        if proof.get('sample_count') != 32 or proof.get('ray_count') != 1003:
            raise ValueError(f'unexpected quadrature or ray count: {name}')
        if name == 'actual_gn_trial':
            if proof.get('expected_trial_count') != 124 or proof.get('observed_trial_count') != 124:
                raise ValueError('actual GN proof must cover the complete 124-trial stream')
            for prefix in ('recorded_stream', 'reference_end'):
                source = Path(proof[prefix+'_path'])
                digest = proof[prefix+'_sha256']
                if not source.is_absolute() or sha(source) != digest:
                    raise ValueError(f'changed GN reference: {source}')
                if recorded_digest(recorded, source) != digest:
                    raise ValueError(f'GN reference not recorded as gate input: {source}')
        if name.endswith('threaded') and (checks.get('threaded_parity') is not True or int(proof['threads']) < 2):
            raise ValueError('threaded gate must actually use multiple threads')
        if sha(folder/'end_status.json') != gate.get('end_status_sha256'):
            raise ValueError(f'gate end digest mismatch: {name}')
    return cert


def transformed_source():
    if sha(EXAMPLE) != ORIGINAL_SHA:
        raise ValueError('original example changed: review a new exact transformation')
    source = EXAMPLE.read_text()
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError('closure anchors must occur exactly once')
    start = source.index(START)
    stop = source.index(END, start)
    old = source[start:stop]
    if old.count('compute_directional_flux_csda(') != 2 or not old.endswith('    end\n'):
        raise ValueError('unexpected original flux-only closure body')
    new = '''    # Audited isolated implementation variant: exact last-value ray cache.
    args.papermatch_fit_mode == "stable" || error("cache variant requires stable fit")
    isempty(args.correction_artifact) && error("cache variant requires frozen artifact")
    cache_scalars(x) = ntuple(i -> getfield(x, i), fieldcount(typeof(x)))
    cache_guard = () -> (
        physics_identity=objectid(physics),
        base=cache_scalars(get_csda_correction()),
        material=cache_scalars(get_csda_material_correction()),
        config=cache_scalars(matcfg), site=cache_scalars(match_site),
        shallow=collect(shallow_flags),
        nodes=[(e.energy,e.charge,e.weight) for e in energy_samples],
        paths=[(p.zenith,p.azimuth,[(s.cell_idx,s.distance) for s in p.segments],
            p.distance_in_volume,p.surface_distance,p.remaining_rock_distance,
            p.surface_exit_z,p.valid) for p in match_paths_v])
    # Physics tables are immutable for this closure lifetime and identity-bound
    # by the outer audited physics-dump/source/runtime certificate.
    ray_cache = GuardedStabilityRayCachePrototype.GuardedRayCache(match_paths_v, cache_guard)
    flux_only_at = (ρ, wv) -> begin
        isequal(ρ,matcfg.rock_density) || error("cache variant requires pinned rock density")
        cfg = mk_cfg(ρ)
        forward = (i,w) -> compute_directional_flux_csda(physics, shallow_flags,
            cfg, match_site, match_paths_v[i], w, energy_samples)
        first(GuardedStabilityRayCachePrototype.evaluate!(ray_cache,wv,forward;
            threaded=args.threaded))
    end
'''
    return source[:start] + new + source[stop:], old, new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--julia', required=True)
    p.add_argument('--certificate', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    args = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    def value(flag):
        if args.count(flag) != 1: raise ValueError(f'exactly one {flag} required')
        return args[args.index(flag)+1]
    if args.count('--papermatch-fit-only') != 1 or value('--papermatch-fit-mode') != 'stable':
        p.error('only stable fit-only is permitted')
    if value('--csda-quadrature') != 'midpoint': p.error('deterministic midpoint quadrature required')
    artifact = Path(value('--correction-artifact')).resolve(strict=True)
    dump = Path(value('--dump')).resolve(strict=True)
    output = Path(value('--output-dir')).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not a.prepare_only:
        if a.certificate is None: p.error('reviewed certificate required')
        cert = verify_certificate(a.certificate)
        for physical in (artifact, dump):
            if cert['certified_context_sha256'].get(str(physical)) != sha(physical):
                p.error(f'uncertified frozen physical context: {physical}')
    source, old, new = transformed_source()
    source_path = output.parent/'cached_example_transformed.jl'
    launcher = output.parent/'cached_entrypoint.jl'
    manifest = output.parent/'implementation_variant.json'
    if any(q.exists() for q in (source_path, launcher, manifest)):
        p.error('fresh variant output required')
    source_path.write_text(source)
    # include_string uses the ORIGINAL filename, preserving relative include and
    # @__DIR__ semantics. PROGRAM_FILE is this launcher, so main is explicit once.
    launcher.write_text('include('+json.dumps(str(HELPER))+')\n'+
        'Base.include_string(Main, read('+json.dumps(str(source_path))+', String), '+
        json.dumps(str(EXAMPLE))+')\nexit(main())\n')
    record = dict(schema_version=1, implementation_variant='guarded_exact_ray_cache_flux_only_v1',
        timing_scope='Different flux-only callback implementation; objective/Jacobian/profile/reduction unchanged',
        original_logical_filename=str(EXAMPLE), original_example_sha256=sha(EXAMPLE),
        helper_path=str(HELPER), helper_sha256=sha(HELPER), entrypoint_sha256=sha(Path(__file__)),
        transformed_source_sha256=sha(source_path), launcher_sha256=sha(launcher),
        replaced_closures=1, original_closure=old, replacement_closure=new,
        frozen_artifact_sha256=sha(artifact), physics_dump_sha256=sha(dump),
        certificate_path=str(a.certificate.resolve()) if a.certificate else None,
        certificate_sha256=sha(a.certificate) if a.certificate else None,
        prepare_only=a.prepare_only, scientific_arguments=args,
        physics_mutation_contract='Physics tables must not mutate during this fit; bound dump/source/runtime and callback lifetime')
    manifest.write_text(json.dumps(record, indent=2)+'\n')
    if a.prepare_only: return 0
    status = subprocess.run([a.julia,'--startup-file=no','--project='+str(ROOT),str(launcher),*args],cwd=ROOT).returncode
    if sha(source_path) != record['transformed_source_sha256'] or sha(launcher) != record['launcher_sha256']:
        raise ValueError('generated Julia source or launcher changed during execution')
    if sha(HELPER) != record['helper_sha256'] or sha(EXAMPLE) != record['original_example_sha256']:
        raise ValueError('implementation changed during execution')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
