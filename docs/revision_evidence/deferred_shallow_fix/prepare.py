from pathlib import Path
import difflib,hashlib,json
root=Path('/home/roland/devspace/diffpumas.jl');out=Path(__file__).parent
p=root/'src/Tomography.jl';source=p.read_text()
old='''        if w <= 1e-12
            return MaterialMixture(matcfg.porous_material), matcfg.porous_density
        end
'''
assert source.count(old)==1
new=source.replace(old,'')
(out/'shallow-mixture.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),new.splitlines(True),fromfile='a/src/Tomography.jl',tofile='b/src/Tomography.jl')))
start=new.index('function mc_cell_mixture_and_density(');end=new.index('\nfunction build_cell_properties_for_mc',start)
candidate=new[start:end].replace('function mc_cell_mixture_and_density(','function mc_cell_mixture_and_density_candidate(',1)
(out/'candidate.jl').write_text(candidate)
files=sorted(set(list((root/'src').rglob('*.jl'))+list((root/'test').rglob('*'))+[root/'Project.toml',root/'Manifest.toml']))
snapshot={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
(out/'frozen-before.json').write_text(json.dumps(snapshot,indent=2)+'\n')
