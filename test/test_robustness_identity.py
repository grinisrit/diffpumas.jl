"""Resume identity fixtures: no Julia calculation or production outputs."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('robust_identity',
    Path(__file__).resolve().parents[1] / 'dev/robustness_run_identity.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RobustnessIdentityTest(unittest.TestCase):
    def identity(self):
        return dict(schema_version=1, dump='/pinned/dump', sha256={'src/Physics.jl': 'physics',
            'dump': 'dump'}, command=['julia', '--startup-file=no', '--project=.', 'example.jl'],
            runtime=dict(version='1.10.12', binary='/julia', sha256='binary'),
            cpus=4, address_space_gib=12, environment={})

    def finish(self, directory, identity):
        for name in m.REQUIRED_OUTPUTS:
            (directory / name).write_text('recorded output\n')
        m.complete(directory, current_identity=identity)

    def test_exact_resume_and_all_identity_changes(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name) / 'scenario'
            identity = self.identity()
            self.assertEqual(m.prepare(directory, identity), 'run')
            # An exact incomplete attempt may be retried.
            self.assertEqual(m.prepare(directory, identity), 'run')
            self.finish(directory, identity)
            self.assertEqual(m.prepare(directory, identity), 'skip')
            original = (directory / 'completed.json').read_bytes()
            for field in ('sha256', 'command', 'runtime', 'cpus', 'environment'):
                changed = copy.deepcopy(identity)
                if field == 'sha256': changed[field]['dump'] = 'new dump'
                elif field == 'command': changed[field] += ['--new-setting']
                elif field == 'runtime': changed[field]['sha256'] = 'new binary'
                elif field == 'cpus': changed[field] = 2
                else: changed[field]['PUMAS_MATERIALS_PATH'] = '/other'
                with self.subTest(field=field), self.assertRaises(ValueError):
                    m.prepare(directory, changed)
                self.assertEqual((directory / 'completed.json').read_bytes(), original)
            # Output corruption cannot be silently skipped or re-certified.
            (directory / m.REQUIRED_OUTPUTS[0]).write_text('changed')
            with self.assertRaises(ValueError): m.prepare(directory, identity)
            with self.assertRaises(ValueError): m.complete(directory, current_identity=identity)

    def test_unknown_marker_and_missing_outputs(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            marker = directory / 'completed.json'
            marker.write_text('{"completed_utc":"historical"}')
            with self.assertRaises(ValueError): m.prepare(directory, self.identity())
            self.assertFalse((directory / 'run_identity.json').exists())
            self.assertEqual(json.loads(marker.read_text()), {'completed_utc': 'historical'})
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            identity = self.identity()
            m.prepare(directory, identity)
            with self.assertRaises(ValueError): m.complete(directory, current_identity=identity)
            self.assertFalse((directory / 'completed.json').exists())
            changed = copy.deepcopy(identity)
            changed['sha256']['src/Physics.jl'] = 'changed mid-run'
            for output in m.REQUIRED_OUTPUTS: (directory / output).write_text('output')
            with self.assertRaises(ValueError): m.complete(directory, current_identity=changed)
            self.assertFalse((directory / 'completed.json').exists())

    def test_source_and_dump_changes_are_hashed(self):
        # Exercise the real source collector, with a separate tiny dump fixture.
        with tempfile.TemporaryDirectory() as name:
            dump = Path(name) / 'physics.pumas'
            dump.write_bytes(b'first')
            before = m.build_identity(m.ROOT, dump, ['julia', '--project=.'], 4, 12,
                                      runtime={'version': 'fixture'})
            self.assertIn('examples/lvd_muography.jl', before['sha256'])
            self.assertIn('dev/run_bounded.py', before['sha256'])
            self.assertIn('dev/robustness_run_identity.py', before['sha256'])
            dump.write_bytes(b'other')
            after = m.build_identity(m.ROOT, dump, ['julia', '--project=.'], 4, 12,
                                     runtime={'version': 'fixture'})
            self.assertNotEqual(m.canonical_digest(before), m.canonical_digest(after))


if __name__ == '__main__':
    unittest.main()
