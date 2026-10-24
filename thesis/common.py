from pathlib import Path
import hashlib, json, os, random, platform, sys
import numpy as np
import time
import subprocess
import tempfile
import contextlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
# Set this before device/environment inspection can initialize CUDA, so the
# recorded environment and subsequent deterministic cuBLAS setup agree.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

def replace_file(source, destination, *, attempts=10):
    """Retry Windows access/sharing locks without deleting the old destination."""
    if attempts < 1:
        raise ValueError('At least one replacement attempt is required.')
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in {5, 32, 33}:
                raise
            if attempt == attempts - 1:
                exc.add_note(
                    f'Atomic replace failed after {attempts} attempts: {destination}. '
                    'Check open file handles, read-only attributes and folder permissions. '
                    'The existing destination was not deleted.'
                )
                raise
            time.sleep(min(.05 * 2 ** attempt, 1.0))


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=path.parent, prefix=path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(payload)
        replace_file(tmp, path)
    finally:
        if tmp is not None:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

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
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=5)
        commit = revision.stdout.strip() if revision.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    return dict(
        python=sys.version,
        platform=platform.platform(), 
        torch=torch.__version__,
        numpy=np.__version__,
        pandas=pandas.__version__,
        sklearn=sklearn.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        gpu_dedicated_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        torch_threads=torch.get_num_threads(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        git_commit=commit
    )

def atomic_torch(path, obj):
    import torch
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        tmp = Path(stream.name)
    try:
        torch.save(obj, tmp)
        replace_file(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)

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


class CumulativeTimer:
    """Persist active work across resumptions; exclude time between invocations.

    Checkpoint at progress boundaries. A hard kill may lose time since the
    last checkpoint. The caller holds the lock associated with this work.
    """

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        old = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        self.previous = float(old.get('seconds', 0.0))
        self.sessions = int(old.get('sessions', 0)) + 1
        self.uncertain_sessions = int(old.get('uncertain_sessions', 0)) + int(old.get('status') == 'running')
        self.started = time.perf_counter()
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.last_checkpoint = None
        self.checkpoint()
        return self

    @property
    def session_seconds(self):
        return time.perf_counter() - self.started

    @property
    def seconds(self):
        return self.previous + self.session_seconds

    def checkpoint(self, status='running'):
        # Timing is auxiliary; limit file replacements while always persisting
        # entry and exit. Scientific/model checkpoints keep their old cadence.
        now = time.perf_counter()
        if (status == 'running' and self.last_checkpoint is not None
                and now - self.last_checkpoint < 5.0):
            return
        atomic_json(self.path, dict(
            seconds=self.seconds, session_seconds=self.session_seconds,
            sessions=self.sessions, uncertain_sessions=self.uncertain_sessions,
            status=status, started_utc=self.started_utc,
            updated_utc=datetime.now(timezone.utc).isoformat(),
            note='Active time only. A hard kill may lose time since the last progress checkpoint.'
        ))
        self.last_checkpoint = time.perf_counter()

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.checkpoint('completed')
        else:
            try:
                self.checkpoint('interrupted')
            except OSError as exc:
                value.add_note(f'Additional timing write failure for {self.path}: {exc}')
