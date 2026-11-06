"""Record exact runtime source bytes used by the architecture lab."""
import hashlib
import json
import zipfile
from pathlib import Path

root=Path("runs/architecture_lab_20261106")
manifest={}
with zipfile.ZipFile(root/"tested_runtime.zip","w",zipfile.ZIP_DEFLATED) as z:
    for path in Path("src").rglob("*.py"):
        z.write(path,path.as_posix())
        manifest[path.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
(root/"runtime_sha256.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(json.dumps({"runtime_source_files":len(manifest)}))
