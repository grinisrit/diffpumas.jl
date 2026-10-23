#!/usr/bin/env python3
"""Adapt official PUMAS v1.2 geometry only for explicit inputs, samples and seed.
Run inside dev/run_bounded.py. Build upstream with CMake outside its source tree.
"""
import argparse, hashlib, json, pathlib, subprocess, time, shutil, difflib
COMMIT='529976ef51bd420f37ec63b1d3f615784d35b626'
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=pathlib.Path,required=True); p.add_argument('--build',type=pathlib.Path,required=True)
 p.add_argument('--output',type=pathlib.Path,required=True); a=p.parse_args()
 source,build,out=a.source.resolve(),a.build.resolve(),a.output.resolve()
 dump=out/'materials-c.pumas'
 if dump.exists():raise SystemExit('Refusing to replace '+str(dump))
 out.mkdir(parents=True,exist_ok=True)
 assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==COMMIT
 root=pathlib.Path(__file__).resolve().parents[1]; inputs=root/'examples/data/revision_physics'
 files=[inputs/'materials.xml']+sorted((inputs/'pumas-materials/dedx/muon').glob('*.txt'))
 before={str(f.relative_to(root)):digest(f) for f in files}
 original=source/'examples/pumas/geometry.c';text=original.read_text()
 replacements={
  'const char * dump_file = "examples/data/materials.pumas";':'const char * dump_file = getenv("PUMAS_REFERENCE_DUMP");\n        if (!dump_file) { fprintf(stderr,"PUMAS_REFERENCE_DUMP required\\n"); return 2; }',
  'pumas_context_create(&context, physics, 0);':'pumas_context_create(&context, physics, 0);\n        const unsigned long reference_seed = strtoul(getenv("PUMAS_REFERENCE_SEED") ? getenv("PUMAS_REFERENCE_SEED") : "42", NULL, 10);\n        pumas_context_random_seed_set(context, &reference_seed);',
  'const int n = 10000;':'const int n = atoi(getenv("PUMAS_REFERENCE_SAMPLES") ? getenv("PUMAS_REFERENCE_SAMPLES") : "10000");\n        if (n <= 0) return 2;\n        const int reference_csda = getenv("PUMAS_REFERENCE_CSDA") ? atoi(getenv("PUMAS_REFERENCE_CSDA")) : 0;',
  'locals->magnet[1] = 2E-05;':'locals->magnet[1] = 0.;',
  'locals->magnet[2] = -4E-05;':'locals->magnet[2] = 0.;',
  'if (state.energy < 1E+02 - FLT_EPSILON) {':'if (reference_csda) {\n                                context->mode.energy_loss = PUMAS_MODE_CSDA;\n                                context->mode.scattering = PUMAS_MODE_DISABLED;\n                                context->limit.energy = energy_threshold;\n                        } else if (state.energy < 1E+02 - FLT_EPSILON) {',
  'const double energy_threshold = energy_max * 1E+03;':'const double energy_threshold = getenv("PUMAS_REFERENCE_SOURCE_MAX") ? strtod(getenv("PUMAS_REFERENCE_SOURCE_MAX"), NULL) : 1E+09;',
  'PRIMARY_ALTITUDE - FLT_EPSILON) {':'PRIMARY_ALTITUDE - FLT_EPSILON && state.energy < energy_threshold) {',
  '%.5lE':'%.17lE',
  '(rock_thickness <= 0.) ? 0. : sqrt(((w2 / n) - w * w) / n)':'sqrt(fmax((w2 / n) - w * w, 0.) / n)',
 }
 for old,new in replacements.items():
  assert old in text,old
  text=text.replace(old,new)
 adapted=out/'geometry_configurable.c'; adapted.write_text(text)
 (out/'geometry_adaptations.patch').write_text(''.join(difflib.unified_diff(original.read_text().splitlines(True),text.splitlines(True),fromfile='upstream/examples/pumas/geometry.c',tofile='geometry_configurable.c')))
 original_flux=source/'examples/pumas/flux.c';flux_text=original_flux.read_text();assert flux_text.count('0.10566')==2
 adapted_flux=out/'flux_configurable.c';adapted_flux.write_text(flux_text.replace('0.10566','0.10565839'))
 (out/'flux_adaptations.patch').write_text(''.join(difflib.unified_diff(flux_text.splitlines(True),adapted_flux.read_text().splitlines(True),fromfile='upstream/examples/pumas/flux.c',tofile='flux_configurable.c')))
 library=(build/'libpumas.so').resolve();shutil.copy2(library,out/library.name)
 link=out/'libpumas.so.1'
 if not link.exists():link.symlink_to(library.name)
 binary=out/'pumas-flat-reference'
 command=['cc','-O2','-std=c99','-I'+str(source/'include'),'-I'+str(source/'examples/pumas'),str(adapted),str(adapted_flux),'-L'+str(build),'-Wl,-rpath,$ORIGIN','-lpumas','-lm','-o',str(binary)]
 subprocess.run(command,check=True)
 dumpcmd=[str(build/'example-dump'),str(inputs/'materials.xml'),str(inputs/'pumas-materials/dedx/muon'),str(dump)]
 started=time.monotonic()
 with (out/'physics-creation.log').open('w') as log: subprocess.run(dumpcmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert before=={str(f.relative_to(root)):digest(f) for f in files},'PUMAS modified pinned input tables'
 meta=dict(compiler_version=subprocess.check_output(['cc','--version'],text=True).splitlines()[0],upstream='https://github.com/niess/pumas',commit=COMMIT,version='1.2.3',release_tag='v1.2',source_geometry_sha256=digest(original),source_flux_sha256=digest(original_flux),adapted_flux_sha256=digest(adapted_flux),geometry_patch_sha256=digest(out/'geometry_adaptations.patch'),flux_patch_sha256=digest(out/'flux_adaptations.patch'),adapted_geometry_sha256=digest(adapted),binary_sha256=digest(binary),runtime_library=library.name,c_library_sha256=digest(out/library.name),c_dump_sha256=digest(dump),inputs_sha256=before,compile_command=command,physics_command=dumpcmd,physics_seconds=time.monotonic()-started,settings=dict(cutoff=.05,elastic_ratio=.05,bremsstrahlung='SSR',pair_production='SSR',photonuclear='DRSS',rock_density=2650,primary_altitude=1000,mode_switch_energy=100,magnetic_field_T=[0,0,0],surface_flux_muon_mass_GeV=.10565839),adaptations=['explicit source-energy cutoff via environment (default1e9GeV); zero beyond cutoff','explicit dump via environment','configurable sample count','explicit MT seed','17-digit flux output','report MC sampling error also for open sky','zero air magnetic field to match Julia flat kernel','surface-flux mass constant aligned to Julia; physics library unchanged'],rng_note='Same integer seed and MC count; C and Julia random streams are independent implementations, not paired histories.')
 (out/'c-reference-provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
