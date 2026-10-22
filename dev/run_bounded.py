#!/usr/bin/env python3
"""Run one sequential research command with inherited CPU/virtual-memory bounds.

Example: python dev/run_bounded.py -- julia --project=. examples/revision_suite.jl
Run this inside tmux. Descendants inherit affinity and address-space limits.
The limit is per process: this launcher is intended for a single Julia process,
not concurrent independent workers. BLAS uses one thread to avoid oversubscription.
"""
import argparse
import json
import os
import resource
import socket
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cpus', type=int, default=20)
parser.add_argument('--memory-gib', type=float, default=48)
parser.add_argument('command', nargs=argparse.REMAINDER)
a = parser.parse_args()
command = a.command[1:] if a.command[:1] == ['--'] else a.command
if not command:
    parser.error('a command is required after --')
available = sorted(os.sched_getaffinity(0))
physical_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
if not 1 <= a.cpus <= int(len(available) * 0.30):
    parser.error('CPU allocation must be between 1 and 30% of available CPUs')
mem_bytes = int(a.memory_gib * 1024 ** 3)
if not 0 < mem_bytes <= int(physical_ram * 0.30):
    parser.error('address-space limit must be positive and <=30% of RAM')
cpus = available[:a.cpus]
os.sched_setaffinity(0, cpus)
resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['JULIA_NUM_THREADS'] = str(a.cpus)
os.environ['JULIA_NUM_PRECOMPILE_TASKS'] = '1'
print(json.dumps(dict(host=socket.gethostname(), cpus=cpus,
                     address_space_bytes=mem_bytes, command=command)), flush=True)
os.execvp(command[0], command)
