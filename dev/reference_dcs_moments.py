#!/usr/bin/env python3
"""Generate independent raw-DCS moment fixtures from official PUMAS 1.2.3.
Usage: python3 dev/reference_dcs_moments.py LIBPUMAS OUTPUT.csv PUMAS_SOURCE_C
Integrates dσ/dq with composite Simpson quadrature in log(q), independently
of PUMAS/Julia material-table integration. No physics dump is read or written.
"""
import ctypes as c
import csv
import hashlib
import math
import pathlib
import sys

library = pathlib.Path(sys.argv[1]).resolve()
lib = c.CDLL(str(library))
version = [c.c_int() for _ in range(3)]
lib.pumas_version(*(c.byref(v) for v in version))
assert tuple(v.value for v in version) == (1, 2, 3)
source = pathlib.Path(sys.argv[3])
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_hash == '1fbd2554994ef9bde2a1a5d3207697b17b3e5dc6889296702a246e1697bddd32'
lib.pumas_dcs_get.argtypes = [c.c_int, c.c_char_p, c.POINTER(c.c_void_p)]
raw_type = c.CFUNCTYPE(c.c_double, *([c.c_double] * 5))
Z, A = 11., 22.
NA = 6.02214076e23
rows = []
for process, model in enumerate((b'SSR', b'SSR', b'DRSS')):
    pointer = c.c_void_p()
    assert lib.pumas_dcs_get(process, model, c.byref(pointer)) == 0
    raw = raw_type(pointer.value)
    cases = [(e, .10565839, .1, .5) for e in (10., 100., 1000.)]
    cases += [(100., m, .05, 1.) for m in (.10565839, 1.77682)]
    for energy, mass, xlo, xhi in cases:
        lo, hi = xlo * energy, xhi * energy
        if process == 2:
            nucleon, pion = .5 * (.938272 + .939565), .13957018
            lo = max(lo, pion + .5 * pion**2 / nucleon)
            hi = min(hi, energy + mass - .5 * (nucleon + mass**2 / nucleon))
        else:
            hi = min(hi, energy + mass * (1 - .75 * 1.648721271 * Z**(1/3)))
        estimates = []
        for n in (65536, 131072):
            h = math.log(hi / lo) / n
            moments = [0., 0., 0.]
            for i in range(n + 1):
                q = lo * math.exp(i * h)
                # Use the left-hand limit at a sharp physical endpoint;
                # its value at the single boundary point has zero measure.
                if i == n:
                    q = hi * (1 - 1e-12)
                weight = (1 if i in (0, n) else 4 if i % 2 else 2)
                base = weight * raw(Z, A, mass, energy, q) * q
                for mode in range(3):
                    moments[mode] += base * q**mode
            estimates.append([v * h / 3 * NA / (A * .001) for v in moments])
        for mode in range(3):
            value = estimates[1][mode]
            error = abs(value - estimates[0][mode]) / value
            assert error < 2e-7, (process, energy, mode, error)
            rows.append((process, energy, mass, xlo, xhi, mode, value, error))
with pathlib.Path(sys.argv[2]).open('w') as stream:
    stream.write('# Official niess/pumas v1.2 commit 529976ef51bd420f37ec63b1d3f615784d35b626; library 1.2.3\n')
    stream.write('# library SHA256 ' + hashlib.sha256(library.read_bytes()).hexdigest() + '\n')
    stream.write('# source src/pumas.c SHA256 ' + source_hash + '\n')
    stream.write('# process 0=bremsstrahlung SSR,1=pair SSR,2=photonuclear DRSS; mode=energy moment 0,1,2\n')
    stream.write('# Z=11,A=22,NA=6.02214076e23; masses/bounds in rows; bounds clipped to physical endpoint\n')
    stream.write('# Simpson log(q),131072 intervals; refinement from65536; muon interior and muon/tau hardcut bands\n')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(('process', 'energy', 'mass', 'xlow', 'xhigh', 'mode', 'value', 'refinement_relative_difference'))
    writer.writerows(rows)
print(f'Wrote {len(rows)} moments; max refinement difference {max(r[-1] for r in rows):.3g}')
