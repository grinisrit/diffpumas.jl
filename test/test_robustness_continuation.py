"""Focused filesystem tests; no Julia or research calculation is executed."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dev'))
import continue_revision_robustness as continuation
import robustness_run_identity as identity


class ContinuationTests(unittest.TestCase):
    def fixture(self, tmp, completed=True):
        root = Path(tmp)
        output = root / 'runs'
        output.mkdir()
        dump = root / 'physics.pumas'
        dump.write_text('pinned physics')
        runtime = dict(version='test', binary='/verified/julia', sha256='runtime')
        inputs = {str(dump): identity.digest_file(dump)}
        common = ['/old/shim', '--startup-file=no', '--project=.', 'examples/lvd_tomography.jl',
                  '--dump', str(dump), '--papermatch-fit-only', '--seed', '20261022',
                  '--samples', '32', '--calibration-mc-samples', '8192']
        def build(root_, dump_, command, cpus, memory_gib, runtime=runtime):
            return dict(schema_version=1, dump=str(dump), sha256={str(dump):identity.digest_file(dump)},
                        command=command, cpus=cpus, address_space_gib=memory_gib,
                        runtime=runtime, environment={})
        baseline = build(root, dump, common + ['--output-dir', str(output / 'baseline')], 4, 12)
        identity.prepare(output / 'baseline', baseline)
        self.outputs(output / 'baseline')
        if completed:
            identity.complete(output / 'baseline', current_identity=baseline)
        execution = dict(mode='robustness', output=str(output), expected_source_sha256=inputs)
        (output / 'execution.json').write_text(json.dumps(execution))
        main = root / 'main'; main.mkdir()
        for name in continuation.MAIN_REPORTS:(main/name).write_text('recorded report')
        (main/'execution.json').write_text(json.dumps(dict(mode='main',output=str(main),expected_source_sha256=inputs)))
        (main/'end_status.json').write_text(json.dumps(dict(returncode=0,execution_completed=True,source_unchanged=True,source_changes=[],stage_reports_present={p:True for p in continuation.MAIN_REPORTS})))
        (main/'launcher.log').write_text(''.join('=== ['+s+'] OK timestamp ===\n' for s in ('topo','main','sweep','detection','contrast'))+'ALL SELECTED STAGES DONE timestamp\n')
        args = types.SimpleNamespace(main_run=main, root=output, cpus=20, memory_gib=36., affinity='0-19',
                                     julia=None, preserve_incomplete=False, dry_run=True)
        return root, output, runtime, build, baseline, args

    @staticmethod
    def outputs(directory):
        for name in identity.REQUIRED_OUTPUTS:
            (directory / name).write_text('normalization\n2.0\n' if name.endswith('predictions.csv') else 'recorded\n')

    @contextlib.contextmanager
    def patched(self, root, runtime, build):
        with patch.object(continuation, 'ROOT', root), patch.object(identity, 'ROOT', root), \
             patch.object(continuation, 'live_processes', return_value=[]), \
             patch.object(identity, 'julia_runtime', return_value=runtime), \
             patch.object(identity, 'build_identity', side_effect=build):
            yield

    def test_dry_plan_exact_ten_and_derived_scales_no_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output, runtime, build, base, args = self.fixture(tmp)
            before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
            with self.patched(root, runtime, build), contextlib.redirect_stdout(io.StringIO()) as stream:
                continuation.run(args)
            plan = json.loads(stream.getvalue())
            self.assertEqual([x['scenario'] for x in plan['cases']], [x[0] for x in continuation.SCENARIOS])
            self.assertEqual(plan['cases'][0]['action'], 'skip')
            self.assertEqual(plan['cases'][-2]['numerical_command_tail'][-2:], ['--papermatch-abs-scale', '1.9'])
            self.assertEqual(plan['cases'][-1]['numerical_command_tail'][-2:], ['--papermatch-abs-scale', '2.1'])
            self.assertEqual(plan['cases'][1]['numerical_command_tail'][-2:], ['--seed', '20261023'])
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob('*')))

    def test_reject_tampered_completion_and_changed_numerical_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output, runtime, build, base, args = self.fixture(tmp)
            with self.patched(root, runtime, build):
                (output / 'baseline/lvd_fit_field.csv').write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'changed or missing'):
                    continuation.run(args)
            self.outputs(output / 'baseline')
            wrong = continuation.expected_case(base, base['command'][:-2], output, 'rho2660', ['--rock-density','2600'])
            identity.prepare(output / 'rho2660', wrong)
            with self.patched(root, runtime, build), self.assertRaisesRegex(ValueError, 'do not match'):
                continuation.run(args)

    def test_refuse_live_driver_or_julia_and_incomplete_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output, runtime, build, base, args = self.fixture(tmp, completed=False)
            with self.patched(root, runtime, build), self.assertRaisesRegex(ValueError, 'Incomplete existing'):
                continuation.run(args)
            proc = root / 'proc';proc.mkdir()
            for pid, command in [(11,['bash','dev/run_revision_robustness.sh',str(output)]),
                                 (12,['julia','--output-dir',str(output / 'baseline')]),
                                 (13,['rg','julia '+str(output)])]:
                d=proc/str(pid);d.mkdir();(d/'cmdline').write_bytes(('\0'.join(command)+'\0').encode())
            found=continuation.live_processes(output, proc)
            self.assertEqual({x['pid'] for x in found},{11,12})
            with patch.object(continuation,'live_processes',return_value=found), self.assertRaisesRegex(ValueError,'still alive'):
                continuation.require_stopped(output)

    def test_main_must_be_completed_all_five_stages_and_driver_dead(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output, runtime, build, base, args = self.fixture(tmp)
            main=args.main_run
            end=json.loads((main/'end_status.json').read_text())
            bad=dict(end,returncode=1)
            (main/'end_status.json').write_text(json.dumps(bad))
            with self.patched(root,runtime,build),self.assertRaisesRegex(ValueError,'not completed'):
                continuation.run(args)
            (main/'end_status.json').write_text(json.dumps(end))
            (main/'launcher.log').write_text('ALL SELECTED STAGES DONE timestamp')
            with self.patched(root,runtime,build),self.assertRaisesRegex(ValueError,'five successful'):
                continuation.run(args)
            proc=root/'proc';proc.mkdir();p=proc/'42';p.mkdir()
            (p/'cmdline').write_bytes(b'python3\0/recorded/launch.py\0main\0')
            self.assertEqual([x['pid'] for x in continuation.main_outer_processes({'/recorded/launch.py'},proc)],[42])
            p=proc/'43';p.mkdir()
            (p/'cmdline').write_bytes(('bash\0dev/run_lvd_paper.sh\0--output-dir\0'+str(main)+'\0').encode())
            self.assertEqual([x['pid'] for x in continuation.live_processes(main,proc)],[43])


    def test_affinity_failure_never_completes_or_starts_next_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,output,runtime,build,base,args=self.fixture(tmp)
            args.dry_run=False
            class Child:
                pid=12345
                def wait(self):return 0
            def launch(command,**kwargs):
                directory=Path(command[command.index('--output-dir')+1]);self.outputs(directory)
                return Child()
            with self.patched(root,runtime,build),patch.object(continuation.subprocess,'Popen',side_effect=launch), \
                 patch.object(continuation,'observe_child',return_value=dict(pid=12345,actual_affinity=[0],affinity_matches=False)), \
                 contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'affinity was not verified'):
                continuation.run(args)
            self.assertFalse((output/'baseline_seed2/completed.json').exists())
            self.assertFalse((output/'cutoff350').exists())
            archive=next((output/'continuations').iterdir())
            self.assertFalse(json.loads((archive/'end_status.json').read_text())['execution_completed'])
        for invalid in ['0-2-4','3-1','0,0','-1','']:
            with self.assertRaises(ValueError):continuation.parse_affinity(invalid)


    def test_fake_execution_preserves_incomplete_and_resource_invariant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output, runtime, build, base, args = self.fixture(tmp, completed=False)
            original = (output/'baseline/run_identity.json').read_bytes()
            args.dry_run=False;args.preserve_incomplete=True
            class Child:
                pid=12345
                def wait(self):return 0
            def launch(command, **kwargs):
                directory=Path(command[command.index('--output-dir')+1])
                self.outputs(directory)
                return Child()
            with self.patched(root, runtime, build), patch.object(continuation.subprocess,'Popen',side_effect=launch), \
                 patch.object(continuation.subprocess,'run'), patch.object(continuation,'observe_child',
                    return_value=dict(pid=12345,actual_affinity=list(range(20)),affinity_matches=True)), \
                 contextlib.redirect_stdout(io.StringIO()):
                continuation.run(args)
            archives=list((output/'continuations').iterdir());self.assertEqual(len(archives),1)
            archive=archives[0]
            self.assertEqual((archive/'preserved_incomplete/baseline/run_identity.json').read_bytes(),original)
            end=json.loads((archive/'end_status.json').read_text())
            self.assertTrue(end['execution_completed']);self.assertEqual(len(end['completed_cases']),10)
            for name,_ in continuation.SCENARIOS:
                record=identity.read_record(output/name)
                audit=json.loads((output/name/'continuation_audit.json').read_text())
                self.assertEqual(audit['new_identity_sha256'],record['sha256'])
                self.assertEqual(audit['invariant_sha256'],continuation.invariant(record['identity']))
                self.assertEqual(audit['invariant_sha256'],continuation.invariant(audit['original_identity']))
                self.assertEqual(audit['root_manifest_sha256'],identity.digest_file(archive/'manifest.json'))
                self.assertEqual(record['identity']['cpus'],20)
                identity.prepare(output/name,record['identity'])
            self.assertEqual(identity.read_record(output/'baseline')['identity']['command'][1:],base['command'][1:])
            args.dry_run=True
            with self.patched(root,runtime,build), contextlib.redirect_stdout(io.StringIO()) as stream:
                continuation.run(args)
            self.assertTrue(all(x['action']=='skip' for x in json.loads(stream.getvalue())['cases']))


if __name__=='__main__':unittest.main()
