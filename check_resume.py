"""Read-only checks before resuming an existing final experiment."""
import argparse
import json
from pathlib import Path
from thesis.common import config
from thesis.pipeline import experiment_key


def inspect_resume(c):
    root = Path(c['output_root'])
    experiment = root / 'experiment.json'
    if not experiment.is_file():
        raise FileNotFoundError(f'No existing experiment at {root}; check config/output root.')
    saved = json.loads(experiment.read_text(encoding='utf-8'))
    if saved['key'] != experiment_key(c):
        raise ValueError('Experiment identity differs. Do not delete or edit saved keys; '
                         'check the original config/data/code before resuming.')
    completed = list((root / 'runs').rglob('complete.json'))
    pending = []
    for state_path in sorted((root / 'runs').rglob('optimizer/state.json')):
        run = state_path.parent.parent
        if (run / 'complete.json').exists():
            continue
        state = json.loads(state_path.read_text(encoding='utf-8'))
        folder = state_path.parent
        slot = state['slot']
        population = folder / f'population_{slot}.f32'
        best = folder / f'best_{slot}.npy'
        expected_bytes = state['population'] * state['dimension'] * 4
        if not population.is_file() or population.stat().st_size != expected_bytes or not best.is_file():
            raise FileNotFoundError(f'Committed optimizer files missing/incorrect size: {folder}')
        run_config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        warm = (root / 'baseline_checkpoints' / run_config['sequence_mode'] /
                f'seed_{run_config["seed"]}' / run_config['split']['name'] / 'training_state.pt')
        if not warm.is_file():
            raise FileNotFoundError(f'Shared warm checkpoint missing: {warm}')
        pending.append(dict(run=str(run.relative_to(root)), generation=state['generation'],
                            committed_FE=state['evaluations'], budget=state['budget']))
    return dict(experiment_key=saved['key'], identity='OK', completed_runs=len(completed),
                interrupted_optimizers=pending,
                note='Read-only identity/header/file-presence checks. Weight contents are not fully validated.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/final.json')
    parser.add_argument('--output-root')
    args = parser.parse_args()
    c = config(args.config)
    if args.output_root:
        c['output_root'] = str(Path(args.output_root).expanduser().resolve())
    print(json.dumps(inspect_resume(c), indent=2))


if __name__ == '__main__':
    main()
