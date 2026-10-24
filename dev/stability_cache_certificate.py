#!/usr/bin/env python3
"""Emit end-bound semantic proofs and assemble a reviewable cache certificate.

Measurements assert deterministic parity, not MC uncertainty or fit accuracy.
Approval is a separate explicit coordinator action after independent review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import run_cached_stability_fit as runner

KINDS = ('dry_serial', 'corrected_serial', 'corrected_threaded', 'actual_gn_trial')
CHECKS = ('full_prediction_bitwise', 'residual_bitwise', 'profile_bitwise',
          'objective_bitwise', 'context_guard_passed')


def digest_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate_proof(proof):
    kind = proof.get('gate_kind')
    if proof.get('schema_version') != 1 or kind not in KINDS:
        raise ValueError('unknown proof schema or gate')
    checks = proof.get('checks', {})
    required = list(CHECKS)
    if kind == 'corrected_threaded': required.append('threaded_parity')
    if kind == 'actual_gn_trial': required.append('trial_stream_complete')
    if any(checks.get(key) is not True for key in required):
        raise ValueError('a mandatory parity check did not pass')
    for key in ('scope', 'sample_scope', 'cpu_affinity'):
        if not proof.get(key): raise ValueError('missing proof metadata: '+key)
    for key in ('sample_count', 'case_count', 'ray_count', 'threads'):
        if type(proof.get(key)) is not int or proof[key] <= 0:
            raise ValueError('invalid positive count: '+key)
    if proof['sample_count'] != 32 or proof['ray_count'] != 1003:
        raise ValueError('unexpected deterministic quadrature or covered ray count')
    if kind == 'corrected_threaded' and proof['threads'] < 2:
        raise ValueError('threaded gate did not use multiple threads')
    if kind == 'actual_gn_trial':
        if proof.get('expected_trial_count') != 124 or proof.get('observed_trial_count') != 124:
            raise ValueError('incomplete actual trial stream')
        for prefix in ('recorded_stream', 'reference_end'):
            path = Path(proof[prefix+'_path'])
            if not path.is_absolute() or runner.sha(path) != proof[prefix+'_sha256']:
                raise ValueError('changed trial reference: '+str(path))
    factory = Path(proof['closure_factory_path'])
    if not factory.is_absolute() or runner.sha(factory) != proof['closure_factory_sha256']:
        raise ValueError('changed exact closure factory')
    transformed, _, _ = runner.transformed_source()
    if proof['transformed_source_sha256'] != digest_text(transformed):
        raise ValueError('proof did not use the exact current transformed source')
    return proof


def write_fresh(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    emit = sub.add_parser('emit-proof')
    emit.add_argument('--measurement', type=Path, required=True)
    emit.add_argument('--output', type=Path, required=True)
    build = sub.add_parser('build')
    build.add_argument('--gate-spec', type=Path, required=True,
                       help='JSON mapping of each gate to {path,proof_path}')
    build.add_argument('--context', type=Path, action='append', default=[], required=True)
    build.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'emit-proof':
        proof = tomllib.loads(args.measurement.read_text())
        proof['schema_version'] = 1
        proof['passed'] = True
        validate_proof(proof)
        write_fresh(args.output, proof)
        return
    spec = json.loads(args.gate_spec.read_text())
    gates = {}; identity = None; runtime = None
    for kind in KINDS:
        folder = Path(spec[kind]['path']).resolve(strict=True)
        end = runner.audit(folder)
        gate_runtime = json.loads((folder/'execution.json').read_text())['julia_runtime']
        if runtime is not None and runtime != gate_runtime: raise ValueError('gate runtimes differ')
        runtime = gate_runtime
        relative = spec[kind]['proof_path']
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('proof path must remain inside audited gate')
        proof_path = folder/relative
        proof = validate_proof(json.loads(proof_path.read_text()))
        if proof['gate_kind'] != kind or end['output_sha256'].get(relative) != runner.sha(proof_path):
            raise ValueError('proof identity or end binding failed')
        current = {key:proof[key] for key in ('closure_factory_path','closure_factory_sha256','transformed_source_sha256')}
        if identity is not None and identity != current: raise ValueError('different factories between gates')
        identity = current
        gates[kind] = dict(path=str(folder), proof_path=relative, proof_sha256=runner.sha(proof_path),
                           end_status_sha256=runner.sha(folder/'end_status.json'), passed=True)
    contexts = [runner.ROOT/'Project.toml', runner.ROOT/'Manifest.toml',
                *sorted((runner.ROOT/'src').glob('*.jl')), *args.context]
    certificate = dict(schema_version=1, approved_for_successor_queue=False,
        approval_status='pending independent coordinator review',
        scope='Exact flux-only callback parity on recorded trials; no full-history, accuracy, uncertainty or whole-fit speed claim',
        implementation_sha256={str(p):runner.sha(p) for p in
            (runner.EXAMPLE,runner.HELPER,Path(runner.__file__).resolve())},
        certified_context_sha256={str(p.resolve()):runner.sha(p) for p in contexts},
        parity_gates=gates, julia_runtime=runtime, **identity)
    # Validate every deployment requirement without publishing approval.
    candidate = args.output.with_name(args.output.name+'.validation-only')
    try:
        write_fresh(candidate, dict(certificate,approved_for_successor_queue=True))
        runner.verify_certificate(candidate)
    finally:
        candidate.unlink(missing_ok=True)
    write_fresh(args.output, certificate)


if __name__ == '__main__':
    main()
