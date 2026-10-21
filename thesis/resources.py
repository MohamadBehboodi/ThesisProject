"""Static experiment counts; no allocations and no model resizing."""
from pathlib import Path
import shutil
import torch
from .network import parameter_count
from .optimizers import NAMES, ALIASES


def validate_settings(c):
    for key in ['epochs', 'batch_size', 'population', 'evaluations', 'tar_evaluations', 'vector_chunk']:
        if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] < 1:
            raise ValueError('A positive integer is required for ' + key)
    if c['population'] < 4 or min(c['evaluations'], c['tar_evaluations']) < c['population']:
        raise ValueError('Population must be >=4 and each FE budget must cover population initialization.')
    if len(c['hidden_widths']) != 4 or any(not isinstance(h, int) or h < 1 for h in c['hidden_widths']):
        raise ValueError('Four positive hidden widths are required.')
    allowed = NAMES | set(ALIASES) | {'DLSTM', 'LogisticRegression', 'MajorityClass'}
    if ('DLSTM' not in c['models'] or len(set(c['models'])) != len(c['models'])
            or set(c['models']) - allowed):
        raise ValueError('Choose unique supported models including the DLSTM reference.')
    if (not c['sequence_modes'] or len(set(c['sequence_modes'])) != len(c['sequence_modes'])
            or set(c['sequence_modes']) - {'full_year', 'last100'}):
        raise ValueError('Choose unique supported sequence modes.')
    if c['fitness'] not in {'half_rmse', 'cross_entropy'}:
        raise ValueError('Unknown shared objective.')
    if not c['seeds'] or len(set(c['seeds'])) != len(c['seeds']):
        raise ValueError('Choose unique seeds.')


def experiment_resources(c):
    validate_settings(c)
    n = parameter_count(c['hidden_widths'])
    folds = len(c['validation_years'])
    seeds = len(c['seeds'])
    modes = len(c['sequence_modes'])
    optimizers = [m for m in c['models'] if ALIASES.get(m, m) in NAMES]
    neural = [m for m in c['models'] if m == 'DLSTM' or ALIASES.get(m, m) in NAMES]
    warm_runs = seeds * (modes * folds + 1)
    # DLSTM validation for both modes, hybrids for the selected mode; final all neural methods.
    saved_models = seeds * (modes * folds + len(optimizers) * folds + len(neural))
    vector_bytes = n * 4
    buffers = 2 * c['population'] * vector_bytes if optimizers else 0
    # Model + 2 Adam moments per warm checkpoint, plus model.pt for each neural run.
    artifacts = (3 * warm_runs + saved_models) * vector_bytes
    working_disk = 6 * vector_bytes
    per_optimizer = {
        m: seeds * (folds + 1) * (c['tar_evaluations'] if m == 'TAR_CHOA' else c['evaluations'])
        for m in optimizers
    }
    return dict(
        trainable_parameters=n, one_vector_bytes=vector_bytes,
        one_vector_GiB=vector_bytes / 2**30, hidden_widths=c['hidden_widths'],
        population=c['population'], two_population_buffers_GiB=buffers / 2**30,
        shared_warm_training_runs=warm_runs, saved_neural_models=saved_models,
        projected_retained_weights_GiB=artifacts / 2**30,
        estimated_peak_disk_GiB=(buffers + artifacts + working_disk) / 2**30,
        adam_static_memory_GiB=4 * vector_bytes / 2**30,
        hybrid_FE_by_optimizer=per_optimizer, total_hybrid_FE=sum(per_optimizer.values()),
        epochs=c['epochs'], other_FE=c['evaluations'], TAR_FE=c['tar_evaluations'],
        equal_FE_budget=c['evaluations'] == c['tar_evaluations'],
    )


def preflight_report(c):
    info = experiment_resources(c)
    output = Path(c['output_root'])
    probe = next(p for p in [output, *output.parents] if p.exists())
    disk_free = shutil.disk_usage(probe).free
    cuda_available = torch.cuda.is_available()
    free_gpu = total_gpu = None
    if cuda_available:
        device = torch.device(c['device']) if str(c['device']).startswith('cuda') else torch.device('cuda')
        free_gpu, total_gpu = torch.cuda.mem_get_info(device)
    checks = dict(
        requested_device_available=not str(c['device']).startswith('cuda') or cuda_available,
        estimated_disk_capacity=disk_free >= info['estimated_peak_disk_GiB'] * 2**30,
        # Static memory is a lower bound. Activations/workspace need the measured benchmark.
        static_gpu_memory=not str(c['device']).startswith('cuda') or (
            free_gpu is not None and free_gpu >= info['adam_static_memory_GiB'] * 2**30),
    )
    info.update(
        device=c['device'], cuda_available=cuda_available,
        gpu=torch.cuda.get_device_name(device) if cuda_available else None,
        gpu_free_GiB=free_gpu / 2**30 if free_gpu is not None else None,
        gpu_dedicated_GiB=total_gpu / 2**30 if total_gpu is not None else None,
        output_volume_free_GiB=disk_free / 2**30, checks=checks,
        static_checks_passed=all(checks.values()),
        notes=[
            'Architecture, population and budget are unchanged by preflight.',
            'Static checks do not certify activation/workspace memory or completion time.',
            'Run the development-only benchmark on the Windows server before choosing the final budget.',
            'Disk estimate assumes sequential successful runs; concurrent/interrupted jobs may need more.',
        ],
    )
    return info
