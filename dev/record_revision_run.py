#!/usr/bin/env python3
"""Record code and input identities before starting a bounded paper run."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
p.add_argument("dump", type=Path)
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
paths = list((root / "src").glob("*.jl"))
paths += [root / "Project.toml", root / "Manifest.toml",
          root / "examples/lvd_tomography.jl", root / "dev/run_lvd_paper.sh",
          root / "dev/run_bounded.py", a.dump.resolve()]
paths += list((root / "examples/data/lvd_conf").rglob("*"))
hashes = {str(f.relative_to(root) if f.is_relative_to(root) else f): hashlib.sha256(f.read_bytes()).hexdigest()
          for f in paths if f.is_file()}
record = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              host=socket.gethostname(), command=a.command,
              git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              git_status=subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True),
              sha256=hashes)
a.output.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
(a.output / f"run_manifest_{stamp}.json").write_text(json.dumps(record, indent=2) + "\n")
