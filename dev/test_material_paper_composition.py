"""Synthetic plan fixtures only; never create or execute a numerical run."""
import json
import tempfile
from pathlib import Path
from compose_material_paper_evidence import composition_plan

with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);original=root/'original';main=root/'recovery';remaining=root/'remaining'
    for folder in (original,main,remaining):folder.mkdir()
    (original/'stages/topo').mkdir(parents=True);(original/'stages/main').mkdir()
    artifact='/archived/unit-artifact.toml'
    sources={'src/Test.jl':'a'*64,'Project.toml':'b'*64,'Manifest.toml':'c'*64,artifact:'d'*64}
    runtime={'version':'synthetic-unit-runtime'}
    def write(path,data):path.write_text(json.dumps(data))
    base=dict(source_sha256=sources,julia_runtime=runtime,correction_artifact=artifact,physics_dump='/archived/physics.pumas',configuration={})
    write(original/'execution.json',base)
    write(original/'stages/main/end_status.json',{'execution_completed':False})
    write(original/'stages/topo/execution.json',{**base,'command':['unit-topo']})
    write(main/'execution.json',{**base,'source_sha256':{k:v for k,v in sources.items() if k!=artifact},'command':['python','unit-recovery']})
    write(main/'recovery_provenance.json',{'original_source_sha256':sources})
    remaining_record={**base,'commands':{k:['unit-scientific',k,'--output-dir',str(remaining/'stages'/k/'data')] for k in ('detection','contrast','sweep')}}
    write(remaining/'execution.json',remaining_record)
    manifest,paths,commands,_=composition_plan(original,main,remaining)
    assert manifest['scope']=='paper_evidence_composition_not_numerical_run'
    assert manifest['original_failed_main_execution_completed'] is False
    assert commands['main']==['python','unit-recovery']
    assert paths['main']==main and commands['contrast']==remaining_record['commands']['contrast']
    assert not (remaining/'stages').exists() # planning must not create pending stages
    changed={**remaining_record,'source_sha256':{**sources,'src/Test.jl':'0'*64}}
    write(remaining/'execution.json',changed)
    try:composition_plan(original,main,remaining)
    except ValueError:pass
    else:raise AssertionError('different core source accepted')
print('synthetic composition plan: actual wrapper commands, transitive artifact, pending paths/no fabricated ends, core mismatch rejection: PASS')
