"""Effective quota parsing and real inherited-bound checks; no research workload."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bounded',ROOT/'dev/run_bounded.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
GIB=1024**3


class BoundedTest(unittest.TestCase):
    def test_v2_ancestor_and_affinity_limits(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);mount=root/'cgroup mount';child=mount/'parent'/'child';child.mkdir(parents=True)
            for folder,cpu,memory in ((mount,'max 100000','max'),
                    (child.parent,'600000 100000',str(20*GIB)),
                    (child,'1000000 100000',str(80*GIB))):
                (folder/'cpu.max').write_text(cpu);(folder/'memory.max').write_text(memory)
            cg=root/'cgroup';cg.write_text('0::/parent/child\n')
            mi=root/'mountinfo';escaped=str(mount).replace(' ',r'\040')
            mi.write_text(f'1 0 0:1 / {escaped} rw - cgroup2 cgroup rw\n')
            limits=m.cgroup_limits(cg,mi)
            self.assertEqual(limits['cpu_equivalents'],6)
            self.assertEqual(limits['memory_bytes'],20*GIB)
            self.assertEqual(limits['unavailable'],[])
            budget=m.effective_budget(list(range(16)),100*GIB,limits)
            self.assertEqual(budget['max_cpus'],1)
            self.assertEqual(budget['max_address_space_bytes'],6*GIB)
            self.assertEqual(m.effective_budget([0,1],10*GIB,limits)['cpu_equivalents'],2)

    def test_v1_controller_mount_roots_and_unlimited_memory(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);cpu=root/'cpu';mem=root/'memory'
            (cpu/'job').mkdir(parents=True);(mem/'job').mkdir(parents=True)
            (cpu/'cpu.cfs_quota_us').write_text('-1');(cpu/'cpu.cfs_period_us').write_text('100000')
            (cpu/'job/cpu.cfs_quota_us').write_text('7000000');(cpu/'job/cpu.cfs_period_us').write_text('100000')
            (mem/'memory.limit_in_bytes').write_text(str(2**63-4096))
            (mem/'job/memory.limit_in_bytes').write_text(str(170*GIB))
            cg=root/'cgroup';cg.write_text('2:cpu,cpuacct:/tenant/job\n3:memory:/tenant/job\n')
            mi=root/'mountinfo';mi.write_text(f'1 0 0:1 /tenant {cpu} rw - cgroup cgroup rw,cpu,cpuacct\n'
                f'2 0 0:2 /tenant {mem} rw - cgroup cgroup rw,memory\n')
            limits=m.cgroup_limits(cg,mi)
            budget=m.effective_budget(list(range(96)),251*GIB,limits)
            self.assertEqual(budget['max_cpus'],21)
            self.assertEqual(budget['max_address_space_bytes'],51*GIB)

    def test_explicit_fallback(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            limits=m.cgroup_limits(root/'missing',root/'also_missing')
            self.assertTrue(limits['unavailable'])
            budget=m.effective_budget(list(range(20)),100*GIB,limits)
            self.assertEqual(budget['max_cpus'],6)
            self.assertEqual(budget['max_address_space_bytes'],30*GIB)

    def test_child_inherits_affinity_limits_environment_and_exit(self):
        limits=m.cgroup_limits()
        physical=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
        if m.effective_budget(list(os.sched_getaffinity(0)),physical,limits)['max_cpus']<1:
            self.skipTest('30% quota cannot fit one whole affinity CPU')
        command=[sys.executable,str(ROOT/'dev/run_bounded.py'),'--cpus','1','--memory-gib','1','--']
        code='import os,resource,json; print(json.dumps(dict(cpus=len(os.sched_getaffinity(0)),memory=resource.getrlimit(resource.RLIMIT_AS),threads=os.environ["JULIA_NUM_THREADS"],blas=os.environ["OPENBLAS_NUM_THREADS"])))'
        child=subprocess.run(command+[sys.executable,'-c',code],capture_output=True,text=True,check=True)
        header,observed=map(json.loads,child.stdout.splitlines())
        self.assertEqual(observed,dict(cpus=1,memory=[GIB,GIB],threads='1',blas='1'))
        self.assertIn('cgroup',header);self.assertIn('effective_budget',header)
        failure=subprocess.run(command+[sys.executable,'-c','raise SystemExit(7)'],capture_output=True)
        self.assertEqual(failure.returncode,7)


if __name__=='__main__':unittest.main()
