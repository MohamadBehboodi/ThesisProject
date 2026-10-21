from pathlib import Path
import base64, hashlib, json, re, zlib
from .common import ROOT

def emit_capsule(paths):
    root=ROOT
    files={}
    for base in paths:
        base=Path(base)
        candidates=[base] if base.is_file() else list(base.rglob('*'))
        for p in candidates:
            if not p.is_file() or p.suffix not in {'.json','.csv','.txt','.md'}:
                continue
            
            if p.name in {'state.json'} or 'physical_evaluations' in p.name:
                continue

            try:
                name=p.resolve().relative_to(root).as_posix()
            except ValueError:
                # An external server volume is restored into a safe relative mirror.
                name='external_paths/'+p.resolve().as_posix().lstrip('/').replace(':','_')

            b=p.read_bytes()
            if len(b)>5_000_000:
                continue

            files[name]=dict(
                sha256=hashlib.sha256(b).hexdigest(),
                data=base64.b64encode(b).decode()
            )

    body=json.dumps(dict(
        version=1,
        files=files
    ),ensure_ascii=False).encode()

    payload=base64.b64encode(zlib.compress(body,9)).decode()
    checksum=hashlib.sha256(body).hexdigest()
    print('THESIS_CAPSULE_BEGIN '+checksum)
    for i in range(0,len(payload),120):
        print(payload[i:i+120])

    print('THESIS_CAPSULE_END')
    print(f'Embedded {len(files)} essential files. Save notebook WITH outputs. Model weights/population buffers are not embedded.')

# This exact function source is also embedded in the last cell of every notebook.
def restore_from_notebooks(notebook_dir, destination):
    from pathlib import Path
    import base64,hashlib,json,re,zlib
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    restored=[]
    for nb in sorted(Path(notebook_dir).rglob('*.ipynb')):
        notebook=json.loads(nb.read_text(encoding='utf-8'))
        texts=[]
        for cell in notebook.get('cells',[]):
            pieces=[]
            for output in cell.get('outputs',[]):
                text=output.get('text',output.get('data',{}).get('text/plain',''))
                pieces.append(''.join(text) if isinstance(text,list) else text)
            texts.append(''.join(pieces))

        for text in texts:
            for match in re.finditer(r'THESIS_CAPSULE_BEGIN ([a-f0-9]{64})\s+([A-Za-z0-9+/=\s]+?)THESIS_CAPSULE_END',text):
                raw=zlib.decompress(base64.b64decode(''.join(match[2].split())))
                if hashlib.sha256(raw).hexdigest()!=match[1]:
                    raise ValueError('Capsule checksum mismatch')
                
                archive=json.loads(raw)
                for name,item in archive['files'].items():
                    p=(destination/name).resolve()
                    if not p.is_relative_to(destination):
                        raise ValueError('Unsafe capsule path')
                    
                    b=base64.b64decode(item['data'])
                    if hashlib.sha256(b).hexdigest()!=item['sha256']:
                        raise ValueError('File checksum mismatch')
                    
                    if p.exists() and p.read_bytes()!=b:
                        # Preserve both historical snapshots; do not silently overwrite conflicting evidence.
                        p=p.with_name(p.name+'.snapshot_'+item['sha256'][:12])

                    p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_bytes(b)
                    restored.append(str(p))
                    
    print(f'Restored {len(set(restored))} files. Weights require model.pt/checkpoint backups.')
    return sorted(set(restored))
