"""Behavioral regression checks for the dev reconstruction."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from thesis.common import config, seed_all, CumulativeTimer
from thesis.data import FEATURES, batches
from thesis.metrics import evaluate, threshold_from_oof
from thesis.network import DLSTM, vector, objective, Fitness, fit_baseline
from thesis.optimizers import run_optimizer, chimp_coefficients, chimp_proposal, search_bounds, NAMES
from thesis.pipeline import lock, validate, select_mode, freeze_protocol, run_final
from thesis.resources import experiment_resources
from thesis.benchmark import benchmark, training_rows
from thesis.analysis import analyze


def fixture_config(root):
    rows = []
    for year in [1389, 1390, 1392, 1394, 1395]:
        for company in range(4):
            for day in range(3):
                date = pd.Timestamp('2010-04-01') + pd.Timedelta(days=365 * (year - 1389) + day)
                row = {name: 1.0 + company * .1 + day * .01 for name in FEATURES}
                row.update({'نماد': f'C{company}', 'نماد-سال': f'C{company}-{year}', 'سال': year,
                            'تاریخ میلادی': int(date.strftime('%Y%m%d')), 'سود یا زیان': company % 2})
                rows.append(row)
    csv = root / 'financial.csv'
    pd.DataFrame(rows).to_csv(csv, index=False)
    c = config('configs/smoke.json')
    c.update(data_path=str(csv), output_root=str(root / 'results'), min_days=2,
             validation_years=[1392], test_years=[1394, 1395], bootstrap_repeats=20)
    return c


class ThresholdTests(unittest.TestCase):
    def test_high_and_low_probability_thresholds(self):
        y = [0, 0, 1, 1]
        for p in [[.96, .97, .98, .99], [.001, .002, .003, .004]]:
            t = threshold_from_oof(y, p)
            self.assertEqual(evaluate(y, p, t)['balanced_accuracy'], 1)
        self.assertGreater(threshold_from_oof(y, [.96, .97, .98, .99]), .95)
        self.assertLess(threshold_from_oof(y, [.001, .002, .003, .004]), .05)

    def test_exact_search_matches_bruteforce_with_ties_and_endpoints(self):
        rng = np.random.default_rng(47)
        for _ in range(35):
            y = rng.integers(0, 2, 20)
            y[:2] = [0, 1]
            p = rng.choice([0., .001, .2, .5, .97, 1.], size=20)
            candidates = np.unique(np.r_[0., .5, p, np.nextafter(p, np.inf)])
            expected = max(balanced_accuracy_score(y, p >= t) for t in candidates)
            actual = balanced_accuracy_score(y, p >= threshold_from_oof(y, p))
            self.assertEqual(actual, expected)

    def test_threshold_tie_is_nearest_half(self):
        self.assertEqual(threshold_from_oof([0, 1], [.7, .7]), .5)

    def test_invalid_inputs_fail_explicitly(self):
        for y, p in [([], []), ([0, 2], [.1, .9]), ([0, 1], [.1, np.nan]), ([1, 1], [.1, .9])]:
            with self.assertRaises(ValueError):
                threshold_from_oof(y, p)

    def test_loss_class_metrics_expose_majority_failure(self):
        m = evaluate([0, 1, 1, 1], [1., 1., 1., 1.])
        self.assertEqual(m['accuracy'], .75)
        self.assertEqual(m['balanced_accuracy'], .5)
        self.assertEqual(m['loss_class_recall'], 0)
        self.assertEqual(m['negative_count'], 1)


class OperatorTests(unittest.TestCase):
    def test_group_schedules_have_declared_endpoints_and_are_nonlinear(self):
        np.testing.assert_allclose(chimp_coefficients(0, 100), [2.5] * 4)
        np.testing.assert_allclose(chimp_coefficients(1, 100), [0] * 4, atol=1e-12)
        middle = chimp_coefficients(.5, 100)
        self.assertEqual(len(np.unique(middle.round(8))), 4)
        self.assertFalse(np.allclose(middle, 1.25))

    def test_normal_and_weighted_updates_by_hand(self):
        x = np.zeros(3)
        leaders = [np.full(3, v) for v in [1., 2., 3., 4.]]
        # f=0 eliminates random A; exact expected four-leader combinations.
        normal, chaotic = chimp_proposal(x, leaders, 0, .7, np.random.default_rng(2), 10, 2,
                                         chaotic_probability=0)
        weighted, _ = chimp_proposal(x, leaders, 0, .7, np.random.default_rng(2), 10, 2,
                                     weighted=True, chaotic_probability=0)
        self.assertFalse(chaotic)
        np.testing.assert_allclose(normal, 2.5)
        np.testing.assert_allclose(weighted, 3.0)  # sum(v^2)/sum(v)
        scaled, _ = chimp_proposal(x, leaders, 0, .7, np.random.default_rng(2), 10, 2,
                                   lam=.5, chaotic_probability=0)
        np.testing.assert_allclose(scaled, 1.25)

    def test_chaotic_branch_is_bounded_coordinatewise_and_reproducible(self):
        args = (np.zeros(20), [np.zeros(20)] * 4, 1.5, .7)
        a, flag = chimp_proposal(*args, np.random.default_rng(8), 2., 7, chaotic_probability=1)
        b, _ = chimp_proposal(*args, np.random.default_rng(8), 2., 7, chaotic_probability=1)
        self.assertTrue(flag)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(a).all() and (np.abs(a) <= 2).all())
        self.assertGreater(len(np.unique(a)), 1)

    def test_warm_start_preserved_and_strict_bounds_rejected(self):
        c = config('configs/smoke.json')
        c.update(_seed=9, evaluations=c['population'])
        initial = np.full(11, 2., dtype=np.float32)
        observed = []
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run_optimizer('CHOA', lambda v: observed.append(v.copy()) or float(np.square(v).sum()),
                          initial, c, Path(tmp))
        np.testing.assert_array_equal(observed[0], initial)
        bound, info = search_bounds(initial, c)
        self.assertGreaterEqual(bound, 2.)
        self.assertFalse(info['warm_start_clipped'])
        with self.assertRaises(ValueError):
            search_bounds(initial, dict(c, weight_bound_policy='error'))

    def test_tar_only_evaluates_current_phase_across_boundary(self):
        c = config('configs/smoke.json')
        c.update(_seed=6, tar_evaluations=23, chimp_chaotic_probability=0)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run_optimizer('TAR_CHOA', lambda v: float(np.square(v).sum()), np.full(11, .2), c, Path(tmp))
            ledger = [json.loads(s) for s in (Path(tmp) / 'physical_evaluations.jsonl').read_text().splitlines()]
        phases = [r for r in ledger if r['proposal'].startswith('tar_lambda')]
        self.assertTrue(any(r['proposal'] == 'tar_lambda1' for r in phases))
        self.assertTrue(any(r['proposal'] == 'tar_lambda2' for r in phases))
        for row in phases:
            before = (row['committed_attempt_index'] - 1) / c['tar_evaluations']
            self.assertEqual(row['proposal'], 'tar_lambda1' if before < .5 else 'tar_lambda2')
        self.assertEqual(len(ledger), 23)

    def test_opposition_operates_even_in_one_dimension(self):
        c = config('configs/smoke.json')
        c.update(_seed=3, evaluations=10, population=4)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run_optimizer('GSOBL_CHOA', lambda v: float(np.square(v - .5).sum()), np.array([-.5]), c, Path(tmp))
            d = json.loads((Path(tmp) / 'diagnostics.json').read_text())
        self.assertGreater(d['counters']['opposition_attempts'], 0)
        self.assertGreater(d['counters']['opposition_accepted'], 0)

    def test_replay_matches_uninterrupted_for_every_optimizer(self):
        for name in sorted(NAMES):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                c = config('configs/smoke.json')
                c.update(_seed=33, evaluations=29, tar_evaluations=29)
                initial = np.full(17, .2)
                def objective(v):
                    return float(np.square(v).sum())
                with contextlib.redirect_stdout(io.StringIO()):
                    expected, _ = run_optimizer(name, objective, initial, c, Path(a))
                calls = 0
                def interrupted(v):
                    nonlocal calls
                    calls += 1
                    if calls == 12:
                        raise RuntimeError('audit interruption')
                    return objective(v)
                with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
                    run_optimizer(name, interrupted, initial, c, Path(b))
                with contextlib.redirect_stdout(io.StringIO()):
                    actual, history = run_optimizer(name, objective, initial, c, Path(b))
                np.testing.assert_array_equal(expected, actual)
                self.assertEqual(history[-1]['evaluations'], 29)
                self.assertGreater(len((Path(b) / 'physical_evaluations.jsonl').read_text().splitlines()), 29)
                # All maps must be closed, including after interruption (native Windows).
                for f in Path(b).glob('population_*.f32'):
                    f.unlink()


class TrainingTests(unittest.TestCase):
    def test_adam_and_global_fitness_use_shared_objective(self):
        rng = np.random.default_rng(7)
        xs = [rng.normal(size=(n, 18)).astype(np.float32) for n in [3, 5, 2]]
        ys = np.array([0, 1, 0])
        for loss in ['half_rmse', 'cross_entropy']:
            c = config('configs/smoke.json')
            c.update(fitness=loss, batch_size=2)
            seed_all(9)
            model = DLSTM(18, [5, 4, 3, 2], dropout=0).eval()
            x, lengths, y = next(batches(xs, ys, 3))
            expected = float(objective(model, model(x, lengths), y, c)[0].detach())
            fitness = Fitness(model, xs, ys, c, torch.device('cpu'))
            actual = fitness(vector(model))
            self.assertAlmostEqual(actual, expected, places=6)
            self.assertAlmostEqual(actual, fitness.last_components['data_loss'] + fitness.last_components['l1_penalty'])

    def test_adam_resume_restores_weights_and_rng(self):
        rng = np.random.default_rng(4)
        xs = [rng.normal(size=(3, 18)).astype(np.float32) for _ in range(4)]
        ys = np.array([0, 1, 0, 1])
        c = config('configs/smoke.json')
        c.update(epochs=2, batch_size=4)
        def make():
            seed_all(13)
            return DLSTM(18, [5, 4, 3, 2])
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            with contextlib.redirect_stdout(io.StringIO()):
                expected = fit_baseline(make(), xs, ys, c, torch.device('cpu'), Path(a))
            calls = 0
            def interrupt():
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError('interrupt second epoch')
            with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
                fit_baseline(make(), xs, ys, c, torch.device('cpu'), Path(b), progress=interrupt)
            with contextlib.redirect_stdout(io.StringIO()):
                actual = fit_baseline(make(), xs, ys, c, torch.device('cpu'), Path(b))
            np.testing.assert_array_equal(vector(expected), vector(actual))
            self.assertTrue(all(p.grad is None for p in actual.parameters()))
            self.assertEqual(json.loads((Path(b) / 'training_timing.json').read_text())['sessions'], 2)

    def test_timer_accumulates_active_work_and_flags_uncertain_previous_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'time.json'
            p.write_text(json.dumps({'seconds': 12.0, 'sessions': 1, 'status': 'running'}))
            with CumulativeTimer(p) as timer:
                self.assertGreaterEqual(timer.seconds, 12.)
                timer.checkpoint()
            saved = json.loads(p.read_text())
            self.assertGreaterEqual(saved['seconds'], 12.)
            self.assertEqual(saved['uncertain_sessions'], 1)
            self.assertEqual(saved['sessions'], 2)
            self.assertEqual(saved['status'], 'completed')


class IntegrationTests(unittest.TestCase):
    def test_native_lock_rejects_another_process_then_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.lock'
            # Pass the path as argv, avoiding Python/shell quoting of Windows paths.
            code = 'import sys; from thesis.pipeline import lock\nwith lock(sys.argv[1]): pass\n'
            with lock(path):
                blocked = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True)
            released = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn('Cannot acquire lock', blocked.stderr)
            self.assertEqual(released.returncode, 0, released.stderr)

    def test_complete_fixture_pipeline_all_main_comparators(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            c = fixture_config(Path(tmp))
            validate(c, ['DLSTM'], c['sequence_modes'])
            select_mode(c)
            validate(c, c['models'])
            freeze_protocol(c)
            run_final(c)
            out = analyze(c)
            summary = pd.read_csv(out / 'model_comparison.csv')
            self.assertEqual(set(summary.model), set(c['models']))
            self.assertIn('loss_class_recall', summary)
            majority = summary[summary.model == 'MajorityClass'].iloc[0]
            self.assertEqual(majority.balanced_accuracy, .5)
            self.assertEqual(len(list((Path(c['output_root']) / 'runs').rglob('complete.json'))), 19)
            self.assertFalse(list(Path(c['output_root']).rglob('population_*.f32')))

    def test_benchmark_uses_training_cohorts_and_separate_reports(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            c = fixture_config(Path(tmp))
            c['benchmark'] = dict(warmup_steps=0, training_steps=1, fitness_repeats=1,
                                  host_population=4, host_generations=1, checkpoint_io=True)
            folder = benchmark(c)
            report = json.loads((folder / 'benchmark.json').read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(len(report['cases']), 4)
            self.assertEqual(len(report['host_costs']), 6)
            self.assertEqual(len(report['schedule_scenarios']), 2)
            for case in report['cases']:
                self.assertFalse(any(s.endswith('-1394') or s.endswith('-1395') for s in case['training_sample_ids']))
                self.assertFalse(case['evaluation_labels_used'])
            self.assertFalse((Path(c['output_root']) / 'protocol_locked.json').exists())
            self.assertFalse((Path(c['output_root']) / 'runs').exists())
            self.assertFalse((folder / 'temporary').exists())
            self.assertTrue((folder / 'Benchmark_Report.md').exists())

    def test_resource_schedule_includes_ga_woa_and_equal_budget(self):
        c = config()
        r = experiment_resources(c)
        self.assertEqual(r['trainable_parameters'], 144412092)
        self.assertEqual(r['total_hybrid_FE'], 6300000)
        self.assertTrue(r['equal_FE_budget'])
        self.assertEqual(r['shared_warm_training_runs'], 39)
        self.assertEqual(r['saved_neural_models'], 165)

    def test_failed_cuda_benchmark_keeps_report_without_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            c = fixture_config(Path(tmp))
            c['device'] = 'cuda'
            with patch('thesis.pipeline.device_for', side_effect=RuntimeError('CUDA unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
                    benchmark(c)
            reports = list(Path(c['output_root']).glob('benchmarks/*/benchmark.json'))
            self.assertEqual(len(reports), 1)
            saved = json.loads(reports[0].read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['config']['device'], 'cuda')
            self.assertFalse(saved['cases'])
            self.assertFalse((reports[0].parent / 'temporary').exists())


if __name__ == '__main__':
    unittest.main()
