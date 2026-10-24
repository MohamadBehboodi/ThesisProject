"""Atomic-I/O regressions and strict compatibility with the published final profile."""
import contextlib
import errno
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from thesis import common, pipeline
from thesis.common import atomic_json, atomic_torch, config, digest, signature, CumulativeTimer
from thesis.optimizers import run_optimizer
from check_resume import inspect_resume


def windows_error(code=5):
    exc = PermissionError(errno.EACCES, 'simulated Windows sharing/access lock')
    exc.winerror = code
    return exc


def original_key(c):
    """Reconstruct the 6e9725f fingerprint, without using production alias logic."""
    baseline = json.loads((common.ROOT / 'docs/Runtime_Source_Compatibility.json').read_text())
    sources = {}
    for path in (common.ROOT / 'thesis').glob('*.py'):
        style = 'CRLF' if b'\r\n' in path.read_bytes() else 'LF'
        sources[path.name] = baseline['baseline_sources'][path.name][style]
    settings = {k: v for k, v in c.items() if k not in
                {'data_path', 'output_root', 'device', 'bootstrap_repeats'} and not k.startswith('_')}
    return signature(dict(settings=settings, sources=sources, data_sha256=digest(c['data_path'])))


class AtomicIOTests(unittest.TestCase):
    def test_transient_windows_locks_retry_and_keep_old_target_until_success(self):
        real_replace = common.os.replace
        for code in [5, 32, 33]:
            with self.subTest(winerror=code), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / 'timing.json'
                target.write_text('{"old": true}')
                stale = target.with_name(target.name + '.tmp')
                stale.write_text('stale interrupted write')
                seen = []
                def busy_then_replace(source, destination):
                    seen.append(Path(source))
                    self.assertNotEqual(Path(source), stale)
                    # The temporary file is closed, including on native Windows.
                    with open(source, 'rb') as stream:
                        self.assertTrue(stream.read())
                    if len(seen) < 3:
                        self.assertEqual(json.loads(target.read_text()), {'old': True})
                        raise windows_error(code)
                    real_replace(source, destination)
                with patch.object(common.os, 'replace', side_effect=busy_then_replace), \
                        patch.object(common.time, 'sleep') as pause:
                    atomic_json(target, {'new': True})
                self.assertEqual(json.loads(target.read_text()), {'new': True})
                self.assertEqual(pause.call_count, 2)
                self.assertEqual(stale.read_text(), 'stale interrupted write')
                self.assertEqual(list(Path(tmp).glob('timing.json.*.tmp')), [])

    def test_permanent_windows_denial_is_bounded_and_preserves_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'state.json'
            target.write_bytes(b'previous committed state')
            with patch.object(common.os, 'replace', side_effect=windows_error()) as replace, \
                    patch.object(common.time, 'sleep') as pause:
                with self.assertRaises(PermissionError) as caught:
                    atomic_json(target, {'new': True})
            self.assertEqual(replace.call_count, 10)
            self.assertEqual(pause.call_count, 9)
            self.assertAlmostEqual(sum(c.args[0] for c in pause.call_args_list), 5.55)
            self.assertEqual(target.read_bytes(), b'previous committed state')
            self.assertEqual(list(Path(tmp).glob('state.json.*.tmp')), [])
            self.assertIn('not deleted', caught.exception.__notes__[0])

    def test_unrelated_os_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'state.json'
            target.write_text('old')
            with patch.object(common.os, 'replace', side_effect=OSError(errno.ENOSPC, 'disk full')) as replace, \
                    patch.object(common.time, 'sleep') as pause:
                with self.assertRaises(OSError):
                    atomic_json(target, {'new': True})
            self.assertEqual(replace.call_count, 1)
            pause.assert_not_called()
            self.assertEqual(target.read_text(), 'old')

    def test_invalid_json_never_replaces_committed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'state.json'
            target.write_text('old')
            for value, kind in [(float('nan'), ValueError), (object(), TypeError)]:
                with self.assertRaises(kind):
                    atomic_json(target, value)
                self.assertEqual(target.read_text(), 'old')
                self.assertEqual(list(Path(tmp).glob('state.json.*.tmp')), [])

    def test_torch_checkpoint_retries_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'training.pt'
            real_replace = common.os.replace
            attempts = []
            def replace(source, destination):
                attempts.append(str(source))
                if len(attempts) == 1:
                    raise windows_error(32)
                real_replace(source, destination)
            with patch.object(common.os, 'replace', side_effect=replace), patch.object(common.time, 'sleep'):
                atomic_torch(target, {'weights': torch.arange(5)})
            torch.testing.assert_close(torch.load(target, weights_only=True)['weights'], torch.arange(5))
            self.assertEqual(len(attempts), 2)
            self.assertEqual(list(Path(tmp).glob('training.pt.*.tmp')), [])

    def test_timer_throttles_running_writes_but_always_persists_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'timing.json'
            clock = [10.]
            with patch.object(common.time, 'perf_counter', side_effect=lambda: clock[0]), \
                    patch.object(common, 'atomic_json', wraps=atomic_json) as writes:
                with CumulativeTimer(target) as timer:
                    clock[0] = 10.1
                    timer.checkpoint()
                    clock[0] = 14.9
                    timer.checkpoint()
                    self.assertEqual(writes.call_count, 1)
                    clock[0] = 15.1
                    timer.checkpoint()
                    self.assertEqual(writes.call_count, 2)
                    clock[0] = 15.2
                self.assertEqual(writes.call_count, 3)
            saved = json.loads(target.read_text())
            self.assertEqual(saved['status'], 'completed')
            self.assertAlmostEqual(saved['seconds'], 5.2)

    def test_exit_timing_failure_does_not_mask_training_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def save(path, value):
                calls.append(1)
                if len(calls) > 1:
                    raise windows_error()
                atomic_json(path, value)
            with patch.object(common, 'atomic_json', side_effect=save):
                with self.assertRaisesRegex(RuntimeError, 'training failed') as caught:
                    with CumulativeTimer(Path(tmp) / 'timing.json'):
                        raise RuntimeError('training failed')
            self.assertIn('Additional timing write failure', caught.exception.__notes__[0])

    @unittest.skipUnless(sys.platform == 'win32', 'Requires native Windows file-sharing handles')
    def test_native_windows_handle_is_retried_until_released(self):
        import ctypes
        from ctypes import wintypes
        import threading
        import time
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'timing.json'
            target.write_text('{"old": true}')
            # Permit read/write but deliberately deny FILE_SHARE_DELETE.
            handle = kernel.CreateFileW(str(target), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            def release():
                time.sleep(.2)
                kernel.CloseHandle(handle)
            worker = threading.Thread(target=release)
            worker.start()
            try:
                atomic_json(target, {'new': True})
            finally:
                worker.join()
            self.assertEqual(json.loads(target.read_text()), {'new': True})


class ResumeCompatibilityTests(unittest.TestCase):
    def test_exact_published_experiment_key_is_preserved(self):
        c = config('configs/final.json')
        self.assertEqual(pipeline.experiment_key(c), original_key(c))

    def test_crlf_and_lf_aliases_are_explicit_exact_hashes(self):
        import hashlib
        manifest = json.loads((common.ROOT / 'docs/Runtime_Source_Compatibility.json').read_text())
        for name, aliases in manifest['sources'].items():
            lf = (common.ROOT / 'thesis' / name).read_bytes().replace(b'\r\n', b'\n')
            for style, content in [('LF', lf), ('CRLF', lf.replace(b'\n', b'\r\n'))]:
                self.assertEqual(aliases[hashlib.sha256(content).hexdigest()],
                                 manifest['baseline_sources'][name][style])

    def test_changed_settings_data_and_unapproved_sources_still_fail_identity(self):
        c = config('configs/final.json')
        expected = pipeline.experiment_key(c)
        for key, value in [('evaluations', 5001), ('hidden_widths', [128, 64, 32, 30])]:
            self.assertNotEqual(pipeline.experiment_key(dict(c, **{key: value})), expected)
        for target in [Path(c['data_path']).name, 'network.py', 'common.py', 'optimizers.py']:
            def changed(path):
                return 'f' * 64 if Path(path).name == target else digest(path)
            with patch.object(pipeline, 'digest', side_effect=changed):
                self.assertNotEqual(pipeline.experiment_key(c), expected)

    def test_read_only_audit_accepts_legacy_key_and_keeps_saved_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = dict(config('configs/final.json'), output_root=str(root))
            (root / 'experiment.json').write_text(json.dumps({'key': original_key(c)}))
            run = root / 'runs/last100/validation/CHOA/seed_17/val_1399'
            folder = run / 'optimizer'
            folder.mkdir(parents=True)
            (folder / 'state.json').write_text(json.dumps(dict(slot=0, population=30, dimension=11,
                generation=52, evaluations=1590, budget=5000)))
            (folder / 'population_0.f32').write_bytes(bytes(30 * 11 * 4))
            np.save(folder / 'best_0.npy', np.zeros(11, dtype=np.float32))
            (run / 'config.json').write_text(json.dumps(dict(sequence_mode='last100', seed=17,
                                                           split={'name': 'val_1399'})))
            warm = root / 'baseline_checkpoints/last100/seed_17/val_1399/training_state.pt'
            warm.parent.mkdir(parents=True)
            warm.write_bytes(b'presence check only')
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            report = inspect_resume(c)
            self.assertEqual(report['identity'], 'OK')
            self.assertEqual(report['interrupted_optimizers'][0]['committed_FE'], 1590)
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                inspect_resume(dict(c, evaluations=5001))

    def test_failed_timing_progress_resumes_from_committed_generation(self):
        c = dict(config('configs/smoke.json'), _seed=17, evaluations=20)
        initial = np.full(11, .2, dtype=np.float32)
        objective = lambda x: float(np.square(x).sum())
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
                contextlib.redirect_stdout(io.StringIO()):
            expected, _ = run_optimizer('CHOA', objective, initial, c, Path(a))
            calls = []
            def timing_failure():
                calls.append(1)
                if len(calls) == 12:
                    raise windows_error()
            with self.assertRaises(PermissionError):
                run_optimizer('CHOA', objective, initial, c, Path(b), progress=timing_failure)
            saved = json.loads((Path(b) / 'state.json').read_text())
            self.assertGreater(saved['evaluations'], 0)
            actual, history = run_optimizer('CHOA', objective, initial, c, Path(b))
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(history[-1]['evaluations'], 20)


if __name__ == '__main__':
    unittest.main()
