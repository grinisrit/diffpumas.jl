#!/usr/bin/env python3
"""Read-only ZIP/source-commit and all-page PDF text/pixel delivery audit."""
import argparse,hashlib,json,resource,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paper-dir',type=Path,default=ROOT/'diffmuontomoLVD');p.add_argument('--bundle-dir',type=Path);p.add_argument('--source-commit',default='1dd8431d74afbad42e62ed005d301881d93f0176');p.add_argument('--report',type=Path,required=True);a=p.parse_args();paper=a.paper_dir.resolve();bundle=(a.bundle_dir or paper/'revision/submission_package_focused_20261025').resolve()
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
 import fitz
 proofpath=bundle/'source_bundle_proof.json';proof=json.loads(proofpath.read_text());assert proof['source_commit']==a.source_commit and proof['clean_build_verified'] and proof['scratch_source_removed']
 archive=bundle/'manuscript_sources.zip';assert sha(archive.read_bytes())==proof['archive_sha256']
 assert set(proof['documents'])=={'main.tex','supplement.tex'}
 union={}
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None;inside=json.loads(z.read('source_manifest.json'));assert inside['source_commit']==a.source_commit and inside['files']==proof['files']
  assert set(z.namelist())==set(proof['files'])|{'source_manifest.json','BUILD.md'}
  for document,record in proof['documents'].items():
   assert inside['documents'][document]['files']==record['files'] and inside['documents'][document]['build_command']==record['build_command']
   assert document in record['files'];union.update(record['files'])
   for name,h in record['files'].items():assert proof['files'][name]==h
  assert union==proof['files']
  for name,h in proof['files'].items():
   assert not Path(name).is_absolute() and '..' not in Path(name).parts
   assert sha(z.read(name))==h,name
   committed=subprocess.check_output(['git','-C',str(paper),'show',a.source_commit+':'+name]);assert sha(committed)==h,name
  instructions=z.read('BUILD.md').decode();assert 'main.tex' in instructions and 'supplement.tex' in instructions
 checked_logs=[]
 for name in ['clean-build.log','clean-build-supplement.log','inventory-build.log','inventory-build-supplement.log']:
  log=bundle/name;assert log.is_file(),name;text=log.read_text(errors='replace').lower()
  for bad in ['undefined control sequence','undefined references','undefined citations','overfull','float too large']:assert bad not in text,(name,bad)
  checked_logs.append(dict(name=name,sha256=sha(log.read_bytes())))
 documents={};total=0
 for document,record in proof['documents'].items():
  assert record['clean_build_verified'];clean=bundle/record['clean_pdf'];assert sha(clean.read_bytes())==record['clean_pdf_sha256'];authoritative=paper/(Path(document).stem+'.pdf')
  with fitz.open(authoritative) as x,fitz.open(clean) as y:
   assert len(x)==len(y)==record['pages'];assert len(x)==(33 if document=='main.tex' else 48);hashes=[]
   for i,(left,right) in enumerate(zip(x,y),1):
    assert left.get_text()==right.get_text(),(document,i,'text')
    u=left.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False);v=right.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
    assert (u.width,u.height,u.n)==(v.width,v.height,v.n) and u.samples==v.samples,(document,i,'pixels');hashes.append(sha(u.samples))
   assert hashes==record['page_render_sha256'],(document,'proof_render_hashes')
   total+=len(x)
  documents[document]=dict(pages=record['pages'],input_files=len(record['files']),all_page_text_equal=True,all_page_pixels_equal=True,proof_render_hashes_verified=True,authoritative_pdf_sha256=sha(authoritative.read_bytes()),clean_pdf_sha256=record['clean_pdf_sha256'],page_render_sha256=hashes)
 assert total==81 and proof['pages']==33 and proof['clean_pdf_sha256']==proof['documents']['main.tex']['clean_pdf_sha256']
 leftovers=[str(x) for x in bundle.glob('owned-manuscript-source-proof-*')];assert not leftovers
 loose_sources=[str(x) for x in bundle.rglob('*') if x.is_file() and x.suffix in {'.tex','.bib','.cls','.sty','.bst'}];assert not loose_sources
 result=dict(passed=True,source_commit=a.source_commit,bundle_dir=str(bundle),source_files_verified=len(union),two_document_dependencies_verified=True,zip_sha256=proof['archive_sha256'],zip_crc_verified=True,proof_sha256=sha(proofpath.read_bytes()),documents=documents,total_pages=total,render_scale=1,render_color='RGB without alpha',clean_build_logs=checked_logs,no_leftover_source_tree=True,zip_readonly=True,no_extraction_or_compilation=True,checker_sha256=sha(Path(__file__).read_bytes()),scope='Delivery integrity and exact81-page source-build equivalence only; no new simulation, coefficient adoption or physical-accuracy claim.')
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in {'documents','clean_build_logs'}},indent=2))
if __name__=='__main__':main()
