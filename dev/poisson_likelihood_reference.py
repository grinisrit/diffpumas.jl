#!/usr/bin/env python3
"""Known-simple-hypothesis Poisson LLR on previously saved paired counts.
No random generator, transport, reconstruction or exposure search is used.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import tomllib

import numpy as np
import scipy
from scipy.stats import binomtest

ALPHA=0.0013498980316301
PROFILES=('legacy_equal_bin','uniform_directional_area','cosine','azimuth_modulated')
FACTORS=(.5,1.,2.,4.,8.,16.)
FORMAT='little-endian signed int64, draw-major'


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def csvrows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def ordered_bins(records):
    values=[float(row['bin']) for row in records]
    if any(not math.isfinite(value) or not value.is_integer() for value in values):
        raise ValueError('noninteger bin identifier')
    return [int(value) for value in values]


def clean_end(root):
    path=root/'end_status.json';record=json.loads(path.read_text())
    if record.get('returncode')!=0 or any(record.get(key) is not True for key in
            ('execution_completed','source_unchanged','runtime_unchanged')):
        raise ValueError(f'require clean completed source experiment: {root}')
    return record


class AuditedInputs:
    def __init__(self,firstorder,extension):
        self.records={root:clean_end(root) for root in (firstorder,extension)}
        self.adopted={}
        for root in self.records:
            self.note(root/'end_status.json','clean end record')
            self.verify(root/'execution.json')
    def note(self,path,role):
        self.adopted[str(path.resolve())]=dict(sha256=digest(path),role=role)
    def verify(self,path):
        path=path.resolve()
        for root,end in self.records.items():
            if path.is_relative_to(root):
                relative=str(path.relative_to(root));expected=end.get('output_sha256',{}).get(relative)
                if not expected or not path.is_file() or digest(path)!=expected:
                    raise ValueError(f'end-bound input hash mismatch: {path}')
                self.adopted[str(path)]=dict(sha256=expected,owner=str(root),relative_path=relative,
                    end_status_sha256=digest(root/'end_status.json'))
                return path
        raise ValueError(f'input not owned by audited source output: {path}')
    def recheck(self):
        for name,record in self.adopted.items():
            if not Path(name).is_file() or digest(Path(name))!=record['sha256']:
                raise ValueError(f'input changed during LLR evaluation: {name}')


def likelihood_ratio_scores(counts,null_rates,alternative_rates,exposure,chunk_draws=128):
    null_rates=np.asarray(null_rates,dtype=np.float64)
    alternative_rates=np.asarray(alternative_rates,dtype=np.float64)
    if null_rates.ndim!=1 or alternative_rates.shape!=null_rates.shape or counts.ndim!=2 or counts.shape[1]!=len(null_rates):
        raise ValueError('rate/count shape mismatch')
    if not math.isfinite(exposure) or exposure<=0 or chunk_draws!=128:
        raise ValueError('positive exposure and fixed 128-draw chunk required')
    if not np.all(np.isfinite(null_rates)) or not np.all(np.isfinite(alternative_rates)) or np.any(null_rates<=0) or np.any(alternative_rates<=0):
        raise ValueError('this reference requires finite strictly positive hypothesis rates')
    if counts.dtype.kind not in ('i','u'):
        raise ValueError('integer counts required')
    weights=np.log(alternative_rates/null_rates)
    offset=exposure*math.fsum(float(x) for x in alternative_rates-null_rates)
    scores=np.empty(counts.shape[0],dtype=np.float64)
    for start in range(0,len(scores),chunk_draws):
        chunk=counts[start:start+chunk_draws]
        if np.any(chunk<0):raise ValueError('negative Poisson count')
        scores[start:start+len(chunk)]=np.asarray(chunk,dtype=np.float64)@weights-offset
    if not np.all(np.isfinite(scores)):raise ValueError('nonfinite likelihood ratio')
    return scores


def open_counts(path,exposure,draws,bins,audit):
    audit.verify(path);metadata_path=audit.verify(path.with_suffix('.json'))
    metadata=json.loads(metadata_path.read_text())
    if metadata.get('format')!=FORMAT or metadata.get('bins')!=bins or metadata.get('draws')!=draws:
        raise ValueError(f'count metadata shape/format mismatch: {path}')
    if metadata.get('exposure')!=exposure or not isinstance(metadata.get('seed'),int):
        raise ValueError(f'count metadata exposure/seed mismatch: {path}')
    if metadata.get('generator')!='numpy.default_rng PCG64; Poisson variates, no Gaussian substitution':
        raise ValueError(f'unexpected saved generator metadata: {path}')
    if path.stat().st_size!=draws*bins*8:
        raise ValueError(f'count byte size mismatch: {path}')
    return np.memmap(path,mode='r',dtype='<i8',shape=(draws,bins),order='C'),metadata


def interval(k,n):
    result=binomtest(k,n);bounds=result.proportion_ci(confidence_level=.95,method='exact')
    return dict(exceedances=k,draws=n,rate=k/n,ci95_lower=bounds.low,ci95_upper=bounds.high)


def benchmark(firstorder,extension,output):
    firstorder=firstorder.resolve();extension=extension.resolve()
    if output.exists():raise ValueError('fresh output directory required')
    audit=AuditedInputs(firstorder,extension)
    original=firstorder/'results';extended=extension/'results'
    template_path=audit.verify(original/'directional_templates.csv')
    shares_path=audit.verify(original/'exposure_shares.csv')
    planning_path=audit.verify(original/'planning.csv')
    extension_config=audit.verify(extended/'config.toml')
    config=tomllib.loads(extension_config.read_text())
    if config['exposure_factors']!=list(FACTORS) or Path(config['firstorder_results']).resolve()!=original:
        raise ValueError('extension grid or original experiment differs')
    templates=csvrows(template_path);shares=csvrows(shares_path)
    if len(templates)!=2700 or len(shares)!=2700 or ordered_bins(templates)!=list(range(1,2701)) or ordered_bins(shares)!=list(range(1,2701)):
        raise ValueError('template/profile bin identity mismatch')
    if tuple(shares[0])!=('bin',*PROFILES):raise ValueError('profile order/schema changed')
    dry=np.array([float(row['dry_flux']) for row in templates])
    linear_signal=np.array([float(row['linear_signal']) for row in templates])
    finite=np.array([float(row['finite_lens_flux']) for row in templates])
    planning={row['profile']:row for row in csvrows(planning_path)}
    output.mkdir(parents=True);outcomes=[]
    audit.note(Path(__file__).resolve(),'LLR postprocess implementation')
    audit.note(Path(sys.executable).resolve(),'Python executable')
    provenance=dict(scope='Known-simple-hypothesis Poisson reference detector on saved counts, not physical optimality or uncertainty calibration.',
        statistic='L=sum n_m*log(rate_alt_m/rate_null_m)-E*sum(rate_alt_m-rate_null_m)',
        units='Rates are directional acceptance share p_m times directional flux. Mean counts are E*rate_m; E=sum directional integrated exposures. E is not expected count.',
        hypotheses='First order: p*(dry_flux+linear_signal). Finite CSDA: p*finite_lens_flux, a conditional biased-surrogate alternative, not stochastic transport truth.',
        threshold='Separate known-alternative empirical calibration-null quantile at 1-alpha; numpy type-7 linear quantile, strict L>threshold. Calibration null counts are reused exactly, not redrawn.',
        interval_scope='Clopper-Pearson 95% intervals for held-out exceedances conditional on the realized fixed calibration threshold. These are marginal intervals, not simultaneous over 48 outcomes. Threshold uncertainty is not propagated; 3000 held-out nulls cannot certify a true 3-sigma tail.',
        grid_scope='Six predeclared exposure factors; no interpolated/certified 95%-power crossing.',
        alpha=ALPHA,chunk_draws=128,python=platform.python_version(),python_executable=str(Path(sys.executable).resolve()),numpy=np.__version__,scipy=scipy.__version__,
        references=[dict(citation='Ofek and Zackay (2018), AJ 155(4),169',doi='10.3847/1538-3881/aab265',
            scope='Background for low-count Poisson matched filtering; this implementation uses the explicit simple-hypothesis likelihood ratio above.')])
    try:
        for profile in PROFILES:
            p=np.array([float(row[profile]) for row in shares])
            if np.any(p<=0) or not np.isclose(p.sum(),1.,rtol=0,atol=1e-14):raise ValueError('invalid acceptance shares')
            null=p*dry;alternatives={'first_order':p*(dry+linear_signal),'finite_csda':p*finite}
            # The original Julia rates use p*dry+p*signal, preserving its exact rounding.
            alternatives['first_order']=null+p*linear_signal
            reference_exposure=float(planning[profile]['matched_E95'])
            for factor_index,factor in enumerate(FACTORS):
                E=factor*reference_exposure
                old_folder=original/f'{profile}_poisson_{factor}'
                new_folder=extended/f'{profile}_poisson_{factor}'
                reused=factor_index<3
                null_folder=old_folder if reused else new_folder
                null_rate_path=audit.verify(null_folder/('null_rates.txt' if reused else 'calibration_null_rates.txt'))
                if not np.array_equal(np.loadtxt(null_rate_path,ndmin=1),null):raise ValueError('null rate/unit mismatch')
                count_paths=dict(calibration_null=null_folder/'calibration_null.bin',test_null=null_folder/'test_null.bin')
                for hypothesis,alternative in alternatives.items():
                    signal_folder=old_folder if reused and hypothesis=='first_order' else new_folder
                    signal_kind='signal' if reused and hypothesis=='first_order' else 'linear_signal' if hypothesis=='first_order' else 'finite_signal'
                    alternative_path=audit.verify(signal_folder/(signal_kind+'_rates.txt'))
                    if not np.array_equal(np.loadtxt(alternative_path,ndmin=1),alternative):raise ValueError('alternative rate/unit mismatch')
                    populations={**count_paths,'signal':signal_folder/(signal_kind+'.bin')}
                    scores={};metadata={}
                    for population,path in populations.items():
                        counts,meta=open_counts(path,E,600 if population=='signal' else 3000,2700,audit)
                        scores[population]=likelihood_ratio_scores(counts,null,alternative,E)
                        metadata[population]=dict(path=str(path),seed=meta['seed'])
                        del counts
                    threshold=float(np.quantile(scores['calibration_null'],1-ALPHA,method='linear'))
                    null_result=interval(int(np.count_nonzero(scores['test_null']>threshold)),3000)
                    signal_result=interval(int(np.count_nonzero(scores['signal']>threshold)),600)
                    row=dict(profile=profile,hypothesis=hypothesis,exposure_factor=factor,total_E=E,
                        expected_null_counts=E*math.fsum(null),expected_signal_counts=E*math.fsum(alternative),
                        threshold=threshold,calibration_exceedances=int(np.count_nonzero(scores['calibration_null']>threshold)),calibration_draws=3000,
                        null_source='original_firstorder' if reused else 'extension',
                        **{'null_'+key:value for key,value in null_result.items()},
                        **{'signal_'+key:value for key,value in signal_result.items()})
                    outcomes.append(row)
                    name=f'{profile}_{factor}_{hypothesis}'
                    np.savez(output/(name+'_scores.npz'),**scores)
                    (output/(name+'_counts.json')).write_text(json.dumps(metadata,indent=2)+'\n')
        audit.recheck()
        with (output/'poisson_likelihood_reference.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(outcomes[0]));writer.writeheader();writer.writerows(outcomes)
        provenance['input_sha256']=audit.adopted
        provenance['outputs_sha256']={p.name:digest(p) for p in output.iterdir() if p.is_file()}
        provenance['execution_completed']=True
        (output/'provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    except Exception as error:
        (output/'failed.json').write_text(json.dumps(dict(execution_completed=False,error=str(error)),indent=2)+'\n')
        raise
    return outcomes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firstorder',type=Path,required=True);parser.add_argument('--extension',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=benchmark(args.firstorder,args.extension,args.output)
    print(f'Completed {len(result)} saved-count Poisson reference outcomes: {args.output}')
if __name__=='__main__':main()
