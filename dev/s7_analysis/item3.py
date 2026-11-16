import numpy as np
from common import *
d={}
for l in open(WORK+'tmpl.tsv'):
    k,v=l.rstrip('\n').split('\t'); d[k]=np.array(v.split(','),dtype=float)
taus=[1e3,3.16e3,1e4,3.16e4,1e5,1e6,1e8]
print('## Item 3: template-noise diagnostic (field T templates, 2700 valid bins)\n')
print('Source: templates/template_test_X_h0_s0.jls (exact CSDA, arm X) and template_test_M_h16384_s431..438.jls (MC replicates, arms M and Mc; Mc uses the same replicate as M). Counts per bin ~ Poisson(tau * max(template, 0)) (Tomography_protocol.jl:118-128), so expected counts per bin = tau * template. Template values are per unit tau (flux).\n')
print('Median template value per bin (flux units, as in the file): X %.4e ; M replicates b=431..438: '%np.median(d['X'])+', '.join('%.4e'%np.median(d[f'M{b}']) for b in range(431,439))+'\n')
M=np.array([d[f'M{b}'] for b in range(431,439)])
print('Median expected counts per bin, tau * median(template). MC columns: median over the 8 replicates of the per-replicate median (min-max of the 8 in brackets). Also the share of bins with expected counts < 1 and < 10 (median over replicates / X).\n')
print('| tau | X: median expected counts | MC: median expected counts [min,max over replicates] | X: bins < 1 count | X: bins < 10 | MC: bins < 1 | MC: bins < 10 |')
print('|---|---|---|---|---|---|---|')
for t in taus:
    xm=np.median(d['X'])*t; mm=[np.median(M[i])*t for i in range(8)]
    f=lambda a,th: float(np.mean(a*t<th))
    print(f"| {t:g} | {xm:.3g} | {np.median(mm):.3g} [{min(mm):.3g}, {max(mm):.3g}] | {f(d['X'],1):.3f} | {f(d['X'],10):.3f} | {np.median([f(M[i],1) for i in range(8)]):.3f} | {np.median([f(M[i],10) for i in range(8)]):.3f} |")
print()
S=np.array([d[f'S{b}'] for b in range(431,439)])
rel=S/np.maximum(M,1e-300)
print('Template (MC) relative error per bin sigma_mc/template: median per replicate '+', '.join('%.3f %%'%(100*np.median(rel[i])) for i in range(8))+f'; median over replicates {100*np.median([np.median(rel[i]) for i in range(8)]):.3f} % (log_templates.txt: 1.99 %).\n')
# tau* : counting rel error 1/sqrt(tau T) = 0.0199 -> tau = 1/(0.0199^2 T)
e=0.0199
print('Tau at which the counting error of a bin, 1/sqrt(tau * template), equals 1.99 % (tau = 1/(0.0199^2 * template)):\n')
ts=[np.median(1/(e**2*M[i])) for i in range(8)]
print(f'- across bins, median of the per-bin tau: MC replicates {np.median(ts):.3g} (range of the 8 replicates {min(ts):.3g} to {max(ts):.3g}); X template {np.median(1/(e**2*d["X"])):.3g}.')
ts2=[np.median(1/(rel[i]**2*M[i])) for i in range(8)]
print(f'- with each bin\'s own MC error instead of 1.99 %: median over bins {np.median(ts2):.3g} (range of replicates {min(ts2):.3g} to {max(ts2):.3g}).')
print(f'- shorthand: expected counts per bin needed = 1/0.0199^2 = {1/e**2:.0f}; at the median template this is tau = {1/e**2/np.median(M):.3g}.\n')
print('Counting error (median over bins, 1/sqrt(tau * template)) against the 1.99 % template error, by exposure (arm M/Mc replicates, median over replicates):\n')
print('| tau | median counting error | ratio to 1.99 % |'); print('|---|---|---|')
for t in taus:
    ce=np.median([np.median(1/np.sqrt(t*M[i])) for i in range(8)])
    print(f'| {t:g} | {100*ce:.2f} % | {ce/e:.2f} |')
print()
# dev field for comparison
print('For the dev field (D): median template X %.4e; D replicates not dumped.'%np.median(d['D_X']))
