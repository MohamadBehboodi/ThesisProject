"""Bounded server benchmark. It never fits/selects using held-out observations."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import math
import shutil
import time
import uuid
import numpy as np
import torch
from .common import atomic_json, atomic_torch, environment, seed_all
from .data import load_data, Scaler, batches, folds, final_split
from .network import make_model, objective, vector, Fitness
from .optimizers import run_optimizer, NAMES, ALIASES
from .resources import experiment_resources


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def peak_memory(device):
    if device.type != 'cuda':
        return dict(allocated_GiB=None, reserved_GiB=None)
    return dict(allocated_GiB=torch.cuda.max_memory_allocated(device) / 2**30,
                reserved_GiB=torch.cuda.max_memory_reserved(device) / 2**30)


def training_rows(meta, c, split):
    # Do not even request evaluation indices or evaluation labels.
    tr = np.flatnonzero(meta.year.between(c['start_year'], split['train_end']).to_numpy())
    if not len(tr) or meta.y.iloc[tr].nunique() != 2:
        raise ValueError('Benchmark requires a nonempty two-class training split.')
    return tr


def checkpoint_measurement(model, optimizer, folder, device):
    measurements = {}
    for kind, payload in [
        ('training_checkpoint', dict(model=model.state_dict(), optimizer=optimizer.state_dict())),
        ('model_weights', dict(model=model.state_dict())),
    ]:
        path = folder / (kind + '.pt')
        synchronize(device)
        started = time.perf_counter()
        atomic_torch(path, payload)
        synchronize(device)
        write = time.perf_counter() - started
        started = time.perf_counter()
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(loaded['model'])
        synchronize(device)
        read = time.perf_counter() - started
        measurements[kind] = dict(bytes=path.stat().st_size, write_seconds=write, load_and_assign_seconds=read)
        del loaded
        path.unlink()
    return measurements


def benchmark_case(c, mode, split_name, split, settings, folder, device, retain_vector):
    xs, meta = load_data(c, mode)
    tr = training_rows(meta, c, split)
    scaler = Scaler().fit([xs[i] for i in tr])
    train_x = scaler.transform([xs[i] for i in tr])
    train_y = meta.y.to_numpy()[tr]
    # Measure the longest real training batch to check the worst sequence-length memory.
    longest = np.argsort([-len(x) for x in train_x], kind='stable')[:c['batch_size']]
    x, lengths, y = next(batches([train_x[i] for i in longest], train_y[longest], c['batch_size']))
    record = dict(mode=mode, split=split_name, training_end_year=split['train_end'],
                  training_samples=len(tr), training_time_steps=sum(map(len, train_x)),
                  training_sample_ids=meta.sample_id.iloc[tr].tolist(),
                  batch_size=len(y), longest_sequence=int(lengths.max()),
                  benchmark_only=True, evaluation_labels_used=False)
    seed_all(c['seeds'][0])
    model = make_model(c, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'])
    x, y = x.to(device), y.to(device)

    def step():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective(model, model(x, lengths), y, c)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite benchmark training loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(settings['warmup_steps']):
        step()
    synchronize(device)
    times = []
    for _ in range(settings['training_steps']):
        started = time.perf_counter()
        step()
        synchronize(device)
        times.append(time.perf_counter() - started)
    record.update(training_step_seconds=times, training_peak=peak_memory(device))
    if settings['checkpoint_io']:
        record['checkpoint_io'] = checkpoint_measurement(model, optimizer, folder, device)
    model.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    initial = vector(model)
    fitness = Fitness(model, train_x, train_y, c, device)
    fitness_times = []
    for repeat in range(settings['fitness_repeats']):
        print(f'Benchmark {mode}/{split_name}: full training fitness pass {repeat + 1}/'
              f'{settings["fitness_repeats"]} ({len(tr)} samples)', flush=True)
        synchronize(device)
        started = time.perf_counter()
        fitness(initial)
        synchronize(device)
        fitness_times.append(time.perf_counter() - started)
    record.update(status='completed', full_fitness_seconds=fitness_times,
                  fitness_peak=peak_memory(device))
    del fitness, model
    return record, initial if retain_vector else None


def benchmark_host(c, initial, settings, folder):
    """Real dimensions/operators, small disk-backed population, trivial fitness.

    Extrapolation does not reproduce the page-cache pressure of 200 vectors.
    """
    records = []
    for requested in c['models']:
        name = ALIASES.get(requested, requested)
        if name not in NAMES:
            continue
        n = settings['host_population']
        per_generation = 3 * n if name == 'TAR_CHOA' else n + (n - n // 2) if name == 'GSOBL_CHOA' else n
        budget = n + settings['host_generations'] * per_generation
        local = dict(c, population=n, evaluations=budget, tar_evaluations=budget,
                     _seed=c['seeds'][0], retain_populations=True)
        path = folder / ('host_' + name)
        print(f'Benchmark host proposal/storage overhead: {name}, population={n}, dimension={len(initial)}', flush=True)
        started = time.perf_counter()
        run_optimizer(name, lambda v: float(v[0]) ** 2, initial, local, path)
        elapsed = time.perf_counter() - started
        diagnostics = json.loads((path / 'diagnostics.json').read_text(encoding='utf-8'))
        counters = diagnostics['counters']
        build = counters['population_build_seconds']
        records.append(dict(
            optimizer=requested, population=n, evaluations=budget,
            total_seconds=elapsed, initialization_seconds_per_agent=build / n,
            update_and_storage_seconds_per_FE=max(0.0, elapsed - build) / (budget - n),
            note='Trivial fitness, full-dimensional operators. Small-population/cache extrapolation is approximate.',
        ))
        shutil.rmtree(path)
    return records


def _estimate(c, cases, host):
    estimates = []
    good = [x for x in cases if x.get('status') == 'completed']
    by_host = {r['optimizer']: r for r in host}
    resources = experiment_resources(c)
    for chosen in c['sequence_modes']:
        if not any(x['mode'] == chosen for x in good):
            continue
        cache = {}
        for mode in c['sequence_modes']:
            measured = [x for x in good if x['mode'] == mode]
            if not measured:
                continue
            # Avoid claiming precision from two measurements: scale an envelope
            # of measured seconds per true step across first/final training cohorts.
            per_step = [t / x['training_time_steps'] for x in measured for t in x['full_fitness_seconds']]
            train_step = [t for x in measured for t in x['training_step_seconds']]
            cache[mode] = dict(low=min(per_step), high=max(per_step), training=max(train_step),
                               io=next((x['checkpoint_io'] for x in measured if 'checkpoint_io' in x), {}))
        if len(cache) != len(c['sequence_modes']):
            continue
        low = high = warm = 0.0
        selected_fe = {}
        for mode in c['sequence_modes']:
            xs, meta = load_data(c, mode)
            plan = folds(c) + ([final_split(c)] if mode == chosen else [])
            for split in plan:
                tr = training_rows(meta, c, split)
                warm += (math.ceil(len(tr) / c['batch_size']) * cache[mode]['training'] * c['epochs']
                         + cache[mode]['io'].get('training_checkpoint', {}).get('write_seconds', 0) * c['epochs']) * len(c['seeds'])
                if mode != chosen:
                    continue
                steps = sum(len(xs[i]) for i in tr)
                for name in c['models']:
                    if ALIASES.get(name, name) not in NAMES:
                        continue
                    budget = c['tar_evaluations'] if name == 'TAR_CHOA' else c['evaluations']
                    overhead = by_host.get(name, {})
                    host_time = (overhead.get('initialization_seconds_per_agent', 0) * c['population']
                                 + overhead.get('update_and_storage_seconds_per_FE', 0) * (budget - c['population']))
                    repeats = len(c['seeds'])
                    low += repeats * (budget * steps * cache[mode]['low'] + host_time)
                    high += repeats * (budget * steps * cache[mode]['high'] + 2 * host_time)
                    selected_fe[name] = selected_fe.get(name, 0) + repeats * budget
        io = cache[chosen]['io']
        saved_weights = resources['saved_neural_models'] * io.get('model_weights', {}).get('write_seconds', 0)
        # One shared warm-checkpoint access per neural run; includes cached reads.
        cached_load = resources['saved_neural_models'] * io.get('training_checkpoint', {}).get('load_and_assign_seconds', 0)
        fixed = warm + saved_weights + cached_load
        estimates.append(dict(
            chosen_sequence_scenario=chosen, hybrid_FE=selected_fe,
            estimated_total_hours_scenario_low=(low + fixed) / 3600,
            estimated_total_hours_scenario_high=(high + fixed) / 3600,
            estimated_warm_training_hours_conservative=warm / 3600,
            interpretation='Measured-rate scenarios, not a confidence interval or completion guarantee.',
        ))
    return estimates


def _markdown(report):
    lines = ['# Windows Server Benchmark', '',
             'This is a cost/memory diagnostic. No evaluation labels or test predictions are used.', '',
             'Status: ' + report['status'], '',
             '## Hardware and static plan', '',
             '- GPU: ' + str(report['environment']['gpu']),
             '- Dedicated GPU bytes: ' + str(report['environment']['gpu_dedicated_bytes']),
             '- Architecture: ' + str(report['resource_plan']['hidden_widths']),
             '- Full trainable parameters: ' + str(report['resource_plan']['trainable_parameters']),
             '- Full experiment hybrid FEs: ' + str(report['resource_plan']['total_hybrid_FE']), '',
             '## Measured training cases', '',
             '| Mode | Training cohort | Samples | Longest sequence | Mean full fitness seconds | Training peak GiB | Status |',
             '|---|---|---:|---:|---:|---:|---|']
    for case in report['cases']:
        times = case.get('full_fitness_seconds', [])
        mean = f'{np.mean(times):.4f}' if times else 'N/A'
        peak = case.get('training_peak', {}).get('allocated_GiB')
        peak_text = f'{peak:.3f}' if peak is not None else 'N/A'
        lines.append(f'| {case["mode"]} | {case["split"]} | {case.get("training_samples", "N/A")} | '
                     f'{case.get("longest_sequence", "N/A")} | {mean} | {peak_text} | {case["status"]} |')
    lines += ['', '## Schedule scenarios', '',
              '| Selected sequence | Scenario low (hours) | Scenario high (hours) |',
              '|---|---:|---:|']
    for row in report['schedule_scenarios']:
        lines.append(f'| {row["chosen_sequence_scenario"]} | {row["estimated_total_hours_scenario_low"]:.2f} | '
                     f'{row["estimated_total_hours_scenario_high"]:.2f} |')
    lines += ['', '## Interpretation', ''] + ['- ' + note for note in report['limitations']]
    return '\n'.join(lines) + '\n'


def benchmark(c):
    from .pipeline import device_for, lock
    settings = dict(warmup_steps=1, training_steps=3, fitness_repeats=2, host_population=4,
                    host_generations=1, checkpoint_io=True,
                    splits=['first_validation', 'final_training'], modes=c['sequence_modes'])
    settings.update(c.get('benchmark', {}))
    for name in ['training_steps', 'fitness_repeats', 'host_generations']:
        if not isinstance(settings[name], int) or settings[name] < 1:
            raise ValueError('Positive integer benchmark setting required: ' + name)
    if settings['warmup_steps'] < 0 or settings['host_population'] < 4:
        raise ValueError('Invalid benchmark warmup or population.')
    if not settings['modes'] or not set(settings['modes']) <= set(c['sequence_modes']):
        raise ValueError('Benchmark modes must belong to sequence_modes.')
    if not settings['splits'] or not set(settings['splits']) <= {'first_validation', 'final_training'}:
        raise ValueError('Unknown benchmark training cohort.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
    folder = Path(c['output_root']) / 'benchmarks' / stamp
    folder.mkdir(parents=True, exist_ok=True)
    report = dict(
        status='running', settings=settings, config=c, environment=environment(),
        resource_plan=experiment_resources(c), cases=[], host_costs=[], schedule_scenarios=[],
        evaluation_labels_used=False,
        limitations=[
            'Architecture and vector dimensions are unchanged. Only a few Adam steps and full training fitness passes are run.',
            'No validation/test predictions, thresholds, model selection or final protocol are produced.',
            'Training step time uses the longest real training batch; warm-training estimates are conservative.',
            'Host operators use a small population. Large populations can exceed RAM/page cache and be slower than extrapolated.',
            'Disk load timings are warm-cache measurements, not guaranteed cold SSD throughput.',
            'Scenario bounds are sensitivity calculations, not statistical confidence intervals.',
            'LR, majority baseline and final statistical analysis time are not estimated.',
            'Budget choice is manual after reviewing this report; no automatic model or budget reduction.',
        ],
    )
    work = folder / 'temporary'
    work.mkdir()
    def save():
        atomic_json(folder / 'benchmark.json', report)
        (folder / 'Benchmark_Report.md').write_text(_markdown(report), encoding='utf-8')
    save()
    try:
        device = device_for(c)
        minimum = (2 * settings['host_population'] + 6) * report['resource_plan']['one_vector_bytes']
        if shutil.disk_usage(work).free < minimum:
            raise OSError(f'Benchmark needs at least {minimum / 2**30:.2f} GiB free temporary space.')
        with lock(Path(c['output_root']) / '.benchmark.lock'):
            first = folds(c)[0] if 'first_validation' in settings['splits'] else None
            cohort = dict(first_validation=first, final_training=final_split(c))
            initial = None
            for mode in settings['modes']:
                for split_name in settings['splits']:
                    print(f'Benchmark starting {mode}/{split_name}; architecture={c["hidden_widths"]}', flush=True)
                    try:
                        case, weights = benchmark_case(c, mode, split_name, cohort[split_name], settings,
                                                       work, device, initial is None)
                        if initial is None:
                            initial = weights
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                        case = dict(mode=mode, split=split_name, status='failed',
                                    error=type(exc).__name__ + ': ' + str(exc))
                    finally:
                        gc.collect()
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    report['cases'].append(case)
                    save()
            if initial is not None:
                report['host_costs'] = benchmark_host(c, initial, settings, work)
                report['schedule_scenarios'] = _estimate(c, report['cases'], report['host_costs'])
            report['status'] = 'completed' if report['cases'] and all(
                x.get('status') == 'completed' for x in report['cases']) else 'partial_failure'
    except BaseException as exc:
        report['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
        save()
        print('Benchmark report: ' + str(folder / 'Benchmark_Report.md'), flush=True)
    return folder
