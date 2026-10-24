"""Independent weight optimizers with bounded, resumable population search."""
from pathlib import Path
import hashlib
import json
import time
import shutil
import numpy as np
from .common import atomic_json, signature, CumulativeTimer, replace_file

NAMES = {'CHOA', 'WCHOA', 'GSOBL_CHOA', 'TAR_CHOA', 'GA', 'WOA'}
ALIASES = {'GA_reference': 'GA', 'WOA_reference': 'WOA'}
STATE_VERSION = 3


def tar_lambdas(q, theta, r, budget, policy='literal'):
    q = np.clip(q, 0, 1 - 1e-12)
    r = np.clip(r, 1e-8, 1 - 1e-8)
    exponent = np.clip(1 - np.tan((r - .5) * np.pi * theta / budget), -20, 20)
    l1 = float(np.exp(np.clip(exponent * np.log(max(1 - q, 1e-12)), -40, 40)))
    l2 = float(np.exp(np.clip(exponent * np.log(max(2 - 2 * q, 1e-12)), -40, 40)))
    if policy == 'stated_ranges':
        return float(np.clip(l1, 0, 1)), float(np.clip(l2, .5, 1))
    if policy != 'literal':
        raise ValueError(policy)
    return l1, l2


def chimp_coefficients(q, budget):
    """Four ChOA2-inspired shapes, endpoint-normalized to [2.5,0].

    This is a declared reconstruction: remove the source's 0.5 floor while
    retaining logarithmic, cubic, exponential and quadratic group schedules.
    Progress is committed FE / budget rather than generation number.
    """
    q = float(np.clip(q, 0, 1))
    budget = max(2, budget)
    return 2.5 * np.array([
        1 - np.log1p((budget - 1) * q) / np.log(budget),
        1 - q ** 3,
        (np.exp(-16 * q * q) - np.exp(-16)) / (1 - np.exp(-16)),
        (1 - q) ** 2,
    ])


def gauss_step(x):
    a = np.asarray(x, dtype=float)
    safe = np.where(a > 1e-12, a, .7)
    result = np.reciprocal(safe) % 1
    return np.where(result > 1e-12, result, .7)


def chimp_proposal(x, leaders, f, chaos, rng, bound, chunk, *,
                   weighted=False, lam=1.0, chaotic_probability=.5, chaos_iterations=3):
    """Normal four-leader update OR a Gauss-mapped chaotic-position update.

    Independent coordinate seeds and a fixed number of Gauss iterations keep
    the chaotic branch vectorized. Seeds/iterations are explicit choices;
    no scalar chaotic value is broadcast to all neural weights.
    """
    chaotic = bool(rng.random() < chaotic_probability)
    z = np.empty(len(x), dtype=np.float32)
    for j in range(0, len(x), chunk):
        k = min(j + chunk, len(x))
        if chaotic:
            state = rng.uniform(1e-8, 1 - 1e-8, k - j)
            for _ in range(chaos_iterations):
                state = gauss_step(state)
            result = -bound + 2 * bound * state
        else:
            terms = []
            for leader in leaders:
                a = 2 * f * rng.random(k - j) - f
                c = 2 * rng.random(k - j)
                terms.append(lam * leader[j:k] - a * np.abs(c * leader[j:k] - chaos * x[j:k]))
            terms = np.stack(terms)
            if weighted:
                magnitude = np.abs(terms)
                denominator = magnitude.sum(0)
                weights = np.divide(magnitude, denominator, out=np.full_like(magnitude, .25),
                                    where=denominator > 1e-12)
                result = (weights * terms).sum(0)
            else:
                result = terms.mean(0)
        z[j:k] = np.clip(result, -bound, bound)
    return z, chaotic


def stochastic_alternative(x, rng, q, bound, chunk):
    z = x.copy()
    for j in range(0, len(x), chunk):
        k = min(j + chunk, len(x))
        r = np.clip(rng.random(k - j), 1e-8, 1 - 1e-8)
        mask = np.tan(np.pi * (r - .5)) < 1 - q
        z[j:k] = np.where(mask, rng.uniform(-bound, bound, k - j), x[j:k])
    return z


def search_bounds(initial, c):
    requested = float(c['weight_bound'])
    policy = c.get('weight_bound_policy', 'expand_to_initial')
    if not np.isfinite(requested) or requested <= 0 or policy not in {'expand_to_initial', 'error'}:
        raise ValueError('Invalid weight bound or bound policy.')
    maximum, outside = 0.0, 0
    for j in range(0, len(initial), c['vector_chunk']):
        part = initial[j:j + c['vector_chunk']]
        if not np.isfinite(part).all():
            raise ValueError('Warm-start weights must be finite.')
        absolute = np.abs(part)
        maximum = max(maximum, float(absolute.max()))
        outside += int((absolute > requested).sum())
    if outside and policy == 'error':
        raise ValueError(f'{outside} warm-start weights exceed +/-{requested}. '
                         'Choose valid common bounds or expand_to_initial; weights were not clipped.')
    bound = max(requested, float(np.nextafter(maximum, np.inf))) if outside else requested
    return bound, dict(requested_bound=requested, effective_bound=bound, policy=policy,
                       warm_max_abs=maximum, warm_outside_requested_bounds=outside,
                       warm_start_clipped=False)


def vector_change(initial, best, chunk):
    changed, squared, largest = 0, 0.0, 0.0
    for j in range(0, len(initial), chunk):
        delta = best[j:j + chunk].astype(np.float64) - initial[j:j + chunk]
        changed += int(np.count_nonzero(delta))
        squared += float(np.dot(delta, delta))
        largest = max(largest, float(np.abs(delta).max()))
    return dict(changed_parameters=changed, l2_distance=float(np.sqrt(squared)),
                max_abs_change=largest, identical_to_warm_start=changed == 0)


def run_optimizer(name, fitness, initial, c, path, progress=None):
    name = ALIASES.get(name, name)
    if name not in NAMES:
        raise ValueError('Unknown optimizer: ' + str(name))
    initial = np.ascontiguousarray(initial, dtype=np.float32)
    n, d = int(c['population']), len(initial)
    budget = int(c['tar_evaluations'] if name == 'TAR_CHOA' else c['evaluations'])
    chunk = int(c['vector_chunk'])
    if n < 4 or budget < n or not d or chunk < 1:
        raise ValueError('Population >=4, budget >= population, and positive dimension/chunk required.')
    probability = c.get('chimp_chaotic_probability', .5)
    iterations = int(c.get('chaos_iterations', 3))
    if not 0 <= probability <= 1 or iterations < 1:
        raise ValueError('Invalid chaotic branch settings.')
    bound, bounds = search_bounds(initial, c)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = path / 'state.json'
    initial_hash = hashlib.sha256(memoryview(initial)).hexdigest()
    settings = {k: v for k, v in c.items() if k not in
                {'data_path', 'output_root', 'device', 'bootstrap_repeats', 'retain_populations'}}
    identity = signature(dict(version=STATE_VERSION, settings=settings, initial=initial_hash, name=name))
    if not checkpoint.exists() and shutil.disk_usage(path).free < 2 * n * d * 4 + 6 * d * 4:
        raise OSError('Insufficient disk for full population buffers. Choose a larger output volume.')
    live = {}
    with CumulativeTimer(path / 'timing.json') as timer:
        try:
            return _search(name, fitness, initial, c, path, progress, timer, live,
                           n, d, budget, chunk, bound, bounds, initial_hash, identity)
        finally:
            # Close memory maps even on Ctrl+C; important for native Windows cleanup.
            for pop in live.values():
                pop.flush()
                pop._mmap.close()


def _search(name, fitness, initial, c, path, progress, timer, live,
            n, d, budget, chunk, bound, bounds, initial_hash, identity):
    rng = np.random.default_rng(c['_seed'])
    history, theta, chaos = [], np.ones(n), .7
    groups = np.arange(n) % 4
    rng.shuffle(groups)
    evaluations = generation = slot = 0
    counters = dict(agents_updated=0, chaotic_proposals=0, normal_chimp_proposals=0,
                    opposition_attempts=0, opposition_accepted=0, tar_alternative_attempts=0,
                    tar_lambda1_evaluations=0, tar_lambda2_evaluations=0,
                    population_build_seconds=0.0, snapshot_seconds=0.0)
    checkpoint = path / 'state.json'
    ledger = path / 'physical_evaluations.jsonl'
    every = max(1, int(c.get('progress_every_evaluations', 10)))

    def mmap(s, mode):
        a = np.memmap(path / f'population_{s}.f32', dtype='float32', mode=mode, shape=(n, d))
        live[id(a)] = a
        return a

    def score(v, kind, agent):
        nonlocal evaluations
        if evaluations >= budget:
            return None
        started = time.perf_counter()
        value = float(fitness(v))
        if not np.isfinite(value):
            value = 1e30
        evaluations += 1
        record = dict(committed_attempt_index=evaluations, fitness=value,
                      seconds=time.perf_counter() - started, proposal=kind, agent=agent)
        record.update(getattr(fitness, 'last_components', {}))
        with ledger.open('a', encoding='utf-8') as log:
            log.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush()
        # One full fitness pass may be long: persist its elapsed work on every FE.
        timer.checkpoint()
        if progress:
            progress()
        if evaluations == 1 or evaluations % every == 0 or evaluations == budget:
            print(f'{name} FE={evaluations}/{budget} objective={value:.7g}', flush=True)
        return value

    def chimp(x, leaders, agent, lam=1.0):
        f = chimp_coefficients(evaluations / budget, budget)[groups[agent]]
        z, chaotic = chimp_proposal(
            x, leaders, f, chaos, rng, bound, chunk,
            weighted=name in {'WCHOA', 'GSOBL_CHOA'}, lam=lam,
            chaotic_probability=c.get('chimp_chaotic_probability', .5),
            chaos_iterations=c.get('chaos_iterations', 3))
        counters['chaotic_proposals' if chaotic else 'normal_chimp_proposals'] += 1
        return z

    def commit(pop, scores, best, best_score):
        started = time.perf_counter()
        pop.flush()
        best_path = path / f'best_{slot}.npy'
        with best_path.with_suffix('.npy.tmp').open('wb') as stream:
            np.save(stream, best)
        replace_file(best_path.with_suffix('.npy.tmp'), best_path)
        counters['snapshot_seconds'] += time.perf_counter() - started
        atomic_json(checkpoint, dict(
            version=STATE_VERSION, identity=identity, slot=slot, dimension=d, population=n,
            name=name, evaluations=evaluations, generation=generation, scores=scores.tolist(),
            best_score=best_score, initial_fitness=initial_fitness, theta=theta.tolist(),
            groups=groups.tolist(), chaos=chaos, rng=rng.bit_generator.state,
            history=history, budget=budget, counters=counters, bounds=bounds,
        ))
        timer.checkpoint()
        if progress:
            progress()

    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding='utf-8'))
        if state.get('version') != STATE_VERSION or state.get('identity') != identity:
            raise ValueError('Optimizer resume identity mismatch. Use a new experiment.')
        slot, evaluations, generation = state['slot'], state['evaluations'], state['generation']
        scores, theta, groups = np.array(state['scores']), np.array(state['theta']), np.array(state['groups'])
        chaos, history, counters = state['chaos'], state['history'], state['counters']
        rng.bit_generator.state = state['rng']
        pop = mmap(slot, 'r+')
        best, best_score = np.load(path / f'best_{slot}.npy'), state['best_score']
        initial_fitness = state['initial_fitness']
    else:
        print(f'{name}: building {n} full vectors, dimension={d}, bound={bound:.6g}', flush=True)
        started = time.perf_counter()
        pop = mmap(0, 'w+')
        for i in range(n):
            for j in range(0, d, chunk):
                k = min(j + chunk, d)
                if i == 0:
                    pop[i, j:k] = initial[j:k]  # Exact baseline, never clipped.
                    continue
                if c['population_initialization'] == 'warm_normal':
                    z = initial[j:k] + rng.normal(0, c['initial_std'], k - j)
                elif c['population_initialization'] == 'uniform':
                    z = rng.uniform(-bound, bound, k - j)
                else:
                    raise ValueError('Unknown population initialization.')
                pop[i, j:k] = np.clip(z, -bound, bound)
        counters['population_build_seconds'] = time.perf_counter() - started
        scores = np.array([score(pop[i], 'initial', i) for i in range(n)])
        initial_fitness = float(scores[0])
        b = int(np.argmin(scores))
        best, best_score = np.array(pop[b]), float(scores[b])
        history.append(dict(generation=0, evaluations=evaluations, best_fitness=best_score))
        commit(pop, scores, best, best_score)

    while evaluations < budget:
        old, old_scores = pop, scores.copy()
        slot = 1 - slot
        pop = mmap(slot, 'w+')
        scores = old_scores.copy()
        leader_ids = np.argsort(old_scores, kind='stable')[:4]
        leaders = [np.array(old[i]) for i in leader_ids]
        if best_score < old_scores[leader_ids[0]]:
            leaders = [best.copy()] + leaders[:3]
        weakest = set(np.argsort(old_scores, kind='stable')[n // 2:])
        chaos = float(gauss_step(chaos))
        for i in range(n):
            pop[i] = old[i]
            if evaluations >= budget:
                continue
            counters['agents_updated'] += 1
            x, incumbent = np.array(old[i]), float(old_scores[i])
            if name == 'GA':
                parents = []
                for _ in range(2):
                    ids = rng.choice(n, size=3, replace=False)
                    parents.append(np.array(old[int(ids[np.argmin(old_scores[ids])])]))
                candidate = np.empty(d, dtype=np.float32)
                for j in range(0, d, chunk):
                    k = min(j + chunk, d)
                    a = rng.random(k - j)
                    z = a * parents[0][j:k] + (1 - a) * parents[1][j:k]
                    mutation = rng.random(k - j) < c['ga_mutation_probability']
                    candidate[j:k] = np.clip(z + mutation * rng.normal(0, c['ga_mutation_std'], k - j), -bound, bound)
                value = score(candidate, 'ga', i)
            elif name == 'WOA':
                q = evaluations / budget
                a = 2 * (1 - q)
                A, C = 2 * a * rng.random() - a, 2 * rng.random()
                p, ell = rng.random(), rng.uniform(-1, 1)
                target = np.array(best if abs(A) < 1 else old[int(rng.integers(n))])
                candidate = np.empty(d, dtype=np.float32)
                for j in range(0, d, chunk):
                    k = min(j + chunk, d)
                    z = (target[j:k] - A * np.abs(C * target[j:k] - x[j:k]) if p < .5
                         else np.abs(best[j:k] - x[j:k]) * np.exp(ell) * np.cos(2 * np.pi * ell) + best[j:k])
                    candidate[j:k] = np.clip(z, -bound, bound)
                value = score(candidate, 'woa', i)
            else:
                if name == 'GSOBL_CHOA' and i in weakest:
                    counters['opposition_attempts'] += 1
                    opposite = -x  # L + U - x for the common symmetric bounds.
                    opposite_value = score(opposite, 'opposition', i)
                    if opposite_value < incumbent:
                        x, incumbent = opposite, opposite_value
                        counters['opposition_accepted'] += 1
                if name == 'TAR_CHOA':
                    alternative = stochastic_alternative(x, rng, evaluations / budget, bound, chunk)
                    alternative_value = score(alternative, 'tar_alternative', i)
                    counters['tar_alternative_attempts'] += 1
                    if alternative_value < incumbent:
                        x, incumbent = alternative, alternative_value
                    options = [(incumbent, x)]
                    if evaluations < budget:
                        base = chimp(x, leaders, i)
                        options.append((score(base, 'tar_base', i), base))
                    if evaluations < budget:
                        # Phase is determined immediately before this actual FE.
                        q = evaluations / budget
                        phase = 1 if q < .5 else 2
                        lambdas = tar_lambdas(q, theta[i], rng.random(), budget, c['tar_lambda_policy'])
                        weighted = chimp(x, leaders, i, lambdas[phase - 1])
                        value = score(weighted, f'tar_lambda{phase}', i)
                        counters[f'tar_lambda{phase}_evaluations'] += 1
                        options.append((value, weighted))
                    value, candidate = min(options, key=lambda item: item[0])
                    theta[i] = theta[i] / 2 if value < old_scores[i] else theta[i] + 1
                else:
                    candidate, value = x, incumbent
                    if evaluations < budget:
                        proposal = chimp(x, leaders, i)
                        proposal_value = score(proposal, 'weighted_chimp' if name != 'CHOA' else 'chimp', i)
                        if name != 'GSOBL_CHOA' or proposal_value < value:
                            candidate, value = proposal, proposal_value
            if value < best_score:
                best, best_score = candidate.copy(), float(value)
            greedy = name in {'GSOBL_CHOA', 'TAR_CHOA', 'GA'}
            if not greedy or value < old_scores[i]:
                pop[i], scores[i] = candidate, value
        generation += 1
        history.append(dict(generation=generation, evaluations=evaluations, best_fitness=best_score))
        commit(pop, scores, best, best_score)
        atomic_json(path / 'convergence.json', history)
        print(f'{name} generation={generation} FE={evaluations}/{budget} best={best_score:.7g}', flush=True)
        old._mmap.close()
        live.pop(id(old))
        del old, leaders
    atomic_json(path / 'diagnostics.json', dict(
        algorithm=name, algorithm_version=STATE_VERSION, committed_evaluations=evaluations,
        warm_start_sha256=initial_hash, warm_fitness=initial_fitness, best_fitness=best_score,
        training_fitness_improvement=initial_fitness - best_score, bounds=bounds, counters=counters,
        vector_change=vector_change(initial, best, chunk),
        phase_policy='Only the currently active lambda proposal is evaluated or archived.',
        gsobl_policy='Standard opposition on the weakest half; greedy acceptance; independent reconstruction.',
    ))
    return best, history
