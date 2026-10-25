#!/usr/bin/env python3
"""Create a deterministic, small source snapshot and evidence inventory."""
from pathlib import Path
import gzip,hashlib,io,json,tarfile
root=Path(__file__).resolve().parents[1]
e=root/'docs/revision_evidence/convergence_round3_20261025';paper=root/'diffmuontomoLVD'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
history=e/'figure_run/supplement_source_history.json'
h=json.loads(history.read_text());h['final_source']['sha256']=sha(paper/'revision/convergence_round3_results.tex')
h['changes']='Reviewer wording: fitted-ray coverage does not imply identifiability; explicit harmonic-extension derivative T; attribution rounded to99.95%; font-compatible T^top. Figure source/data/graphics unchanged.'
history.write_text(json.dumps(h,indent=2)+'\n')
files=sorted(list((root/'dev').glob('convergence_round3*'))+list((root/'src').glob('*.jl'))+[root/n for n in ['Project.toml','Manifest.toml','AGENTS.md','examples/lvd_tomography.jl','examples/lvd_muography.jl','examples/export_lvd_topography.jl']]+list(e.glob('*preregistration.md'))+list(e.glob('*command.json'))+[paper/'revision/convergence_round3_results.tex',paper/'revision/build_convergence_round3_figure.py',e/'figure_run/render_source_snapshot.tex',history]+list((paper/'revision/convergence_round3_20261025').glob('*')))
payload=io.BytesIO()
with tarfile.open(fileobj=payload,mode='w') as archive:
    for p in files:
        data=p.read_bytes();info=tarfile.TarInfo(str(p.relative_to(root)));info.size=len(data);info.mode=0o644;info.mtime=0;archive.addfile(info,io.BytesIO(data))
with (e/'source_snapshot.tar.gz').open('wb') as stream:
    with gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0) as zipped:zipped.write(payload.getvalue())
(e/'source_snapshot_manifest.json').write_text(json.dumps(dict(archive_sha256=sha(e/'source_snapshot.tar.gz'),members={str(p.relative_to(root)):sha(p) for p in files},recipe_source_sha256=sha(Path(__file__)),scope='Curated immutable source and figure evidence archive, not a checkout; pinned physics dump remains external and hash-bound in execution records.'),indent=2)+'\n')
(e/'evidence_inventory.json').write_text(json.dumps({str(p.relative_to(e)):sha(p) for p in sorted(e.rglob('*')) if p.is_file() and p.name!='evidence_inventory.json'},indent=2)+'\n')
print(json.dumps(dict(archive_sha256=sha(e/'source_snapshot.tar.gz'),paper_tex_sha256=sha(paper/'revision/convergence_round3_results.tex'))))
