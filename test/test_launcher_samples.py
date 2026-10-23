"""Sampling budgets are explicit and validated before research computations."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]

class LauncherSamples(unittest.TestCase):
    def test_main_defaults_overrides_and_invalid(self):
        script=['bash','dev/run_lvd_paper.sh','--dry-run','--stages','main']
        flags={'--samples':32,'--calibration-mc-samples':1024,'--inverse-mc-samples':256,
               '--mc-samples':1024,'--papermatch-mc-samples':512}
        output=subprocess.check_output(script,cwd=ROOT,text=True)
        for flag,count in flags.items():self.assertIn(f'{flag} {count} ',output)
        overrides=[item for flag in flags for item in (flag,'777')]
        output=subprocess.check_output(script+overrides,cwd=ROOT,text=True)
        for flag in flags:self.assertIn(f'{flag} 777 ',output)
        for flag in flags:
            self.assertEqual(subprocess.run(script+[flag,'0'],cwd=ROOT,capture_output=True).returncode,2)

    def test_robustness_environment_and_seed_reference(self):
        source=(ROOT/'dev/run_revision_robustness.sh').read_text()
        # Evaluate the actual command-array setup without running any scenarios.
        prefix=source.split('revision_run baseline\n')[0]
        with tempfile.TemporaryDirectory() as directory:
            dump=Path(directory)/'physics';dump.write_bytes(b'fixture')
            env=dict(os.environ,REVISION_PHYSICS_DUMP=str(dump))
            command=prefix+'\nprintf "%s\\n" "${revision_common[@]}"\n'
            # Script normally resolves its own directory; use absolute invocation origin.
            command=command.replace('cd "$(dirname "${BASH_SOURCE[0]}")/.."',f'cd "{ROOT}"')
            def run(extra):
                return subprocess.run(['bash','-c',command,'fixture',directory],env=dict(env,**extra),text=True,capture_output=True)
            default=run({});self.assertEqual(default.returncode,0,default.stderr)
            self.assertIn('--samples\n32\n',default.stdout)
            self.assertIn('--calibration-mc-samples\n1024\n',default.stdout)
            custom=run({'REVISION_CSDA_SAMPLES':'64','REVISION_CALIBRATION_MC_SAMPLES':'2048'})
            self.assertEqual(custom.returncode,0,custom.stderr)
            self.assertIn('--samples\n64\n',custom.stdout)
            self.assertIn('--calibration-mc-samples\n2048\n',custom.stdout)
            self.assertEqual(run({'REVISION_CSDA_SAMPLES':'0'}).returncode,2)
        self.assertIn('revision_run baseline_seed2 --seed 20261023',source)
        summary=(ROOT/'dev/summarize_revision_robustness.py').read_text()
        self.assertIn("'baseline_seed2'",summary)
        self.assertIn('MC-noise reference (one alternate seed)',summary)

if __name__=='__main__':unittest.main()
