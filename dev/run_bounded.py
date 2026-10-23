#!/usr/bin/env python3
"""Run one command within 30% of effective affinity/host/cgroup budgets.

CPU affinity and per-process address-space limits are inherited by descendants.
This is not an aggregate memory cap for independent worker processes. Cgroup v1
and v2 ancestor quotas are read when visible; missing controllers fall back to
CPU affinity/physical RAM and are explicitly recorded. BLAS uses one thread.
"""
import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import resource
import socket
import sys


def unescape_mount(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)


def cgroup_limits(cgroup_file=Path('/proc/self/cgroup'), mountinfo_file=Path('/proc/self/mountinfo')):
    result=dict(cpu_equivalents=None, memory_bytes=None, observed=[], unavailable=[])
    try:
        memberships={}
        for line in cgroup_file.read_text().splitlines():
            _, controllers, path=line.split(':', 2)
            for controller in controllers.split(','):
                memberships[controller]=PurePosixPath(path)
        mounts=[]
        for line in mountinfo_file.read_text().splitlines():
            before,after=line.split(' - ', 1)
            left,right=before.split(),after.split()
            if right[0] in ('cgroup','cgroup2'):
                mounts.append((right[0],PurePosixPath(unescape_mount(left[3])),
                               Path(unescape_mount(left[4])),set(right[2].split(','))))
    except (OSError,ValueError,IndexError) as error:
        result['unavailable'].append('cgroup discovery: '+str(error))
        return result
    seen=set()
    for kind,root,mount,controllers in mounts:
        for controller in ('cpu','memory'):
            membership=memberships.get('' if kind=='cgroup2' else controller)
            if membership is None or (kind=='cgroup' and controller not in controllers):
                continue
            try:
                current=mount / str(membership.relative_to(root))
            except ValueError:
                continue
            while True:
                files=('cpu.max',) if controller=='cpu' and kind=='cgroup2' else (
                    ('cpu.cfs_quota_us','cpu.cfs_period_us') if controller=='cpu' else
                    ('memory.max',) if kind=='cgroup2' else ('memory.limit_in_bytes',))
                paths=tuple(current / name for name in files)
                if paths not in seen:
                    seen.add(paths)
                    try:
                        values=[p.read_text().strip() for p in paths]
                        limit=None
                        if controller=='cpu':
                            quota,period=values[0].split() if kind=='cgroup2' else values
                            if quota!='max' and int(quota)>0 and int(period)>0:
                                limit=int(quota)/int(period)
                            field='cpu_equivalents'
                        else:
                            if values[0]!='max' and int(values[0])>0:
                                limit=int(values[0])
                            field='memory_bytes'
                        result['observed'].append(dict(controller=controller,files=[str(p) for p in paths],values=values))
                        if limit is not None:
                            old=result[field]
                            result[field]=limit if old is None else min(old,limit)
                    except (OSError,ValueError):
                        pass
                if current==mount:
                    break
                current=current.parent
    for controller,key in (('cpu','cpu_equivalents'),('memory','memory_bytes')):
        if not any(item['controller']==controller for item in result['observed']):
            result['unavailable'].append(controller+' controller unavailable; using host/affinity fallback')
    return result


def effective_budget(available, physical_ram, cgroup):
    cpus=min(len(available),cgroup['cpu_equivalents']) if cgroup['cpu_equivalents'] is not None else len(available)
    memory=min(physical_ram,cgroup['memory_bytes']) if cgroup['memory_bytes'] is not None else physical_ram
    return dict(cpu_equivalents=cpus,memory_bytes=memory,
                max_cpus=math.floor(cpus*.30+1e-12),max_address_space_bytes=int(memory*.30))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpus',type=int,default=16)
    parser.add_argument('--memory-gib',type=float,default=36)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:
        parser.error('a command is required after --')
    available=sorted(os.sched_getaffinity(0))
    physical_ram=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    cgroup=cgroup_limits()
    budget=effective_budget(available,physical_ram,cgroup)
    if not 1<=args.cpus<=budget['max_cpus']:
        parser.error(f"CPU allocation must be 1..{budget['max_cpus']} (30% of effective affinity/cgroup budget)")
    if not math.isfinite(args.memory_gib):
        parser.error('memory limit must be finite')
    memory=int(args.memory_gib*1024**3)
    if not 0<memory<=budget['max_address_space_bytes']:
        parser.error('address-space limit must be positive and <=30% of effective physical/cgroup RAM')
    cpus=available[:args.cpus]
    os.sched_setaffinity(0,cpus)
    resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['JULIA_NUM_THREADS']=str(args.cpus)
    os.environ['JULIA_NUM_PRECOMPILE_TASKS']='1'
    print(json.dumps(dict(host=socket.gethostname(),cpus=cpus,address_space_bytes=memory,
                          effective_budget=budget,cgroup=cgroup,command=command)),flush=True)
    os.execvp(command[0],command)


if __name__=='__main__':main()
