from pathlib import Path
import hashlib, json, os, random, platform, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()

def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]

def config(path='configs/default.json'):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    
    c = json.loads(p.read_text(encoding='utf-8'))
    for key in ['data_path', 'output_root']:
        q = Path(c[key])
        c[key] = str(q if q.is_absolute() else ROOT / q)
    return c

def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def environment():
    import torch, sklearn, pandas
    return dict(
        python=sys.version,
        platform=platform.platform(), 
        torch=torch.__version__,
        numpy=np.__version__,
        pandas=pandas.__version__,
        sklearn=sklearn.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )

def atomic_torch(path, obj):
    import torch
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    torch.save(obj, tmp)
    os.replace(tmp,path)

def rng_state():
    import torch
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )

def restore_rng(state):
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
