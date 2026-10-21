from pathlib import Path
import contextlib, json, os, time, shutil
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from .common import ROOT,atomic_json,atomic_torch,signature,digest,seed_all,environment
from .data import prepare,load_data,folds,final_split,indices,Scaler,summary_features
from .network import make_model,fit_baseline,predict,vector,assign,Fitness,parameter_count
from .optimizers import run_optimizer
from .metrics import evaluate,threshold_from_oof

def experiment_key(c):
    settings={k:v for k,v in c.items() if k not in ['data_path','output_root','device','bootstrap_repeats'] and not k.startswith('_')}
    sources={p.name:digest(p) for p in (ROOT/'thesis').glob('*.py')}
    return signature(
        dict(
            settings=settings,
            sources=sources,
            data_sha256=digest(c['data_path'])
        )
    )

def check_experiment(c):
    root=Path(c['output_root'])
    root.mkdir(parents=True,exist_ok=True)
    key=experiment_key(c)
    p=root/'experiment.json'

    if p.exists() and json.loads(p.read_text())['key']!=key:
        raise ValueError('Code/data/config changed: choose a new output_root. Results must not be mixed.')
    
    if not p.exists():
        atomic_json(p,
                    dict(
                        key=key,
                        config=c,
                        environment=environment()
                    )
                )
        
    return key

@contextlib.contextmanager
def lock(path):
    import os
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a+b") as f:
        if os.name == "nt":
            import msvcrt

            # Windows needs an existing byte to lock.
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                f.write(b"\0")
                f.flush()

            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot acquire lock; another process may be "
                    f"using this run: {path.parent}"
                ) from exc

            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

        else:

            import fcntl

            try:
                fcntl.flock(
                    f.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot acquire lock; another process may be "
                    f"using this run: {path.parent}"
                ) from exc

            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def device_for(c):
    if c['device']=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install a compatible CUDA-enabled PyTorch; final config will not silently fall back to CPU.')
    return torch.device(c['device'])

def run_one(c,model_name,mode,split,seed,threshold=.5):
    key=check_experiment(c)
    path=Path(c['output_root'])/'runs'/mode/split['stage']/model_name/f'seed_{seed}'/split['name']
    path.mkdir(parents=True,exist_ok=True)
    with lock(path/'.lock'):
        if (path/'complete.json').exists():
            old=json.loads((path/'complete.json').read_text())
            if old['key']!=key or old['threshold']!=threshold:
                raise ValueError('Run identity mismatch.')
            
            for file in ['predictions.csv','metrics.json','config.json']:
                if not (path/file).exists():
                    raise FileNotFoundError(f'Restore missing {file} from notebook capsule or choose a new output_root.')
                
            return path
        
        if split['stage']=='test':
            p=Path(c['output_root'])/'protocol_locked.json'
            if not p.exists():
                raise RuntimeError('Run freeze_protocol before final test.')
            
            protocol=json.loads(p.read_text())
            if protocol['key']!=key or protocol['mode']!=mode or model_name not in protocol['thresholds']:
                raise ValueError('Final run does not match frozen protocol.')
            
            if threshold!=protocol['thresholds'][model_name]:
                raise ValueError('Use the frozen threshold.')
            
        xs,meta=load_data(c,mode)
        tr,ev=indices(meta,c,split)
        scaler=Scaler().fit([xs[i] for i in tr])
        train_x=scaler.transform([xs[i] for i in tr])
        eval_x=scaler.transform([xs[i] for i in ev])
        train_y=meta.y.to_numpy()[tr]
        eval_y=meta.y.to_numpy()[ev]
        run_c=dict(c,_seed=seed)

        atomic_json(path/'config.json',
            dict(
                experiment_key=key,
                config=c,
                model=model_name,
                sequence_mode=mode,
                seed=seed,
                split=split,
                threshold=threshold,
                training_samples=len(tr),
                evaluation_samples=len(ev)
            )
        )

        atomic_json(path/'scaler.json',scaler.state())
        atomic_json(path/'sample_ids.json',
            dict(
                train=meta.sample_id.iloc[tr].tolist(),
                evaluation=meta.sample_id.iloc[ev].tolist()
            )
        )

        seed_all(seed);start=time.perf_counter()
        
        if model_name=='LogisticRegression':
            # Separate fixed-dimensional scaler, train-only. Raw transformed sequences preserve feature semantics.
            a=summary_features([xs[i] for i in tr])
            b=summary_features([xs[i] for i in ev])
            lr_scaler=StandardScaler().fit(a)
            lr=LogisticRegression(C=1.0,max_iter=5000,solver='lbfgs',random_state=seed)
            lr.fit(lr_scaler.transform(a),train_y)
            p=lr.predict_proba(lr_scaler.transform(b))[:,1]
            joblib.dump(
                dict(
                    model=lr,
                    scaler=lr_scaler
                ),
            path/'model.joblib')

            atomic_json(path/'lr_coefficients.json',
                dict(
                    coef=lr.coef_.tolist(),
                    intercept=lr.intercept_.tolist(),
                    scaler_mean=lr_scaler.mean_.tolist(),
                    scaler_scale=lr_scaler.scale_.tolist(),
                    feature_rule='10 annual last + 8 daily last + 8 daily mean + 8 daily population std'
                )
            )
            history=[]
        else:
            device=device_for(c)
            model=make_model(c,device)
            shared=Path(c['output_root'])/'baseline_checkpoints'/mode/f'seed_{seed}'/split['name']
            with lock(shared/'.lock'):
                fit_baseline(model,train_x,train_y,c,device,shared)

            if model_name!='DLSTM':
                best,history=run_optimizer(model_name,Fitness(model,train_x,train_y,c,device),vector(model),run_c,path/'optimizer')
                assign(model,best)
            else:
                history=json.loads((shared/'training_history.json').read_text())

            p=predict(model,eval_x,eval_y,c,device)
            atomic_torch(path/'model.pt',
                dict(
                    state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                    hidden_widths=c['hidden_widths'],
                    dropout=c['dropout'],
                    features=18,
                    key=key
                )
            )
    
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        pred=meta.iloc[ev].copy()
        pred['probability']=p
        pred['prediction']=(p>=threshold).astype(int)
        pred['model']=model_name
        pred['seed']=seed
        pred['mode']=mode
        pred['split']=split['name']
        pred['stage']=split['stage']
        pred.to_csv(path/'predictions.csv',index=False)

        atomic_json(path/'metrics.json',evaluate(eval_y,p,threshold))
        atomic_json(path/'history.json',history)
        atomic_json(path/'complete.json',
            dict(
                key=key,
                threshold=threshold,
                seconds=time.perf_counter()-start,
                model=model_name,
                stage=split['stage'],
                seed=seed,
                mode=mode
            )
        )

        if not c['retain_populations'] and (path/'optimizer').exists():
            # Completed runs use model.pt; interrupted runs retain their populations for resume.
            for file in (path/'optimizer').glob('population_*.f32'):file.unlink()
            for file in (path/'optimizer').glob('best_*.npy'):file.unlink()

            atomic_json(path/'optimizer'/'resume_status.json',
                dict(
                    completed=True,
                    populations_removed=True,
                    note='Finished model is model.pt. Population resume applies only to interrupted runs.'
                )
            )
        return path

def validate(c,models,modes=None):
    from .recovery import emit_capsule
    if modes is None:
        modes=[selected_mode(c)]

    result=[]
    for mode in modes:
        for name in models:
            for seed in c['seeds']:
                for split in folds(c):
                    path=run_one(c,name,mode,split,seed)
                    result.append(path);emit_capsule([path])
    return result

def collect_oof(c,name,mode):
    paths=[Path(c['output_root'])/'runs'/mode/'validation'/name/f'seed_{seed}'/f'val_{v}'/'predictions.csv'
        for seed in c['seeds'] for v in c['validation_years']]
    
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f'Missing required validation run: {p}')
        
    return pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)

def select_mode(c):
    key=check_experiment(c)
    rows=[]
    for mode in c['sequence_modes']:
        frame=collect_oof(c,'DLSTM',mode)
        # Each temporal fold/seed equal weight; threshold fixed before sequence selection.
        scores=[evaluate(g.y,g.probability,.5)['balanced_accuracy'] for _,g in frame.groupby(['seed','split'])]
        
        if any(x is None for x in scores):
            raise ValueError('Single-class validation fold: redefine validation years before selection.')
        
        rows.append(
            dict(
                mode=mode,
                mean_balanced_accuracy=float(np.mean(scores)),
                std=float(np.std(scores)),
                runs=len(scores)
            )
        )
    
    # Predeclared tie break favors full year, not whichever happened to run last.
    winner=max(rows,key=lambda r:(r['mean_balanced_accuracy'],r['mode']=='full_year'))['mode']
    decision=dict(
        key=key,
        mode=winner,
        criterion='mean validation fold/seed Balanced Accuracy at threshold 0.5',
        comparisons=rows
    )
    
    p=Path(c['output_root'])/'sequence_choice.json'
    if p.exists() and json.loads(p.read_text())!=decision:
        raise ValueError('Existing sequence decision differs; use new output_root.')
    
    atomic_json(p,decision)
    return decision

def selected_mode(c):
    p=Path(c['output_root'])/'sequence_choice.json'
    if not p.exists():
        raise RuntimeError('Run both baseline sequence modes, then select_mode.')
    
    choice=json.loads(p.read_text())
    if choice['key']!=check_experiment(c):
        raise ValueError('Sequence choice fingerprint mismatch.')
    
    return choice['mode']

def freeze_protocol(c):
    mode=selected_mode(c)
    thresholds={}
    for name in c['models']:
        f=collect_oof(c,name,mode)
        # Ensemble seed predictions: a company-year remains one observation.
        ens=f.groupby('sample_id',sort=True).agg(
            y=('y','first'),
            p=('probability','mean'),
            n=('seed','nunique')
        )

        if not (ens.n==len(c['seeds'])).all():
            raise ValueError('Missing ensemble seed predictions.')
        
        thresholds[name]=threshold_from_oof(ens.y,ens.p)

    decision=dict(
        key=check_experiment(c),
        mode=mode,
        thresholds=thresholds,
        seeds=c['seeds'],
        test_years=c['test_years'],
        test_training_end=final_split(c)['train_end'],
        primary_metric='balanced_accuracy',
        threshold_selection='development OOF seed ensemble only'
    )

    p=Path(c['output_root'])/'protocol_locked.json'
    if p.exists() and json.loads(p.read_text())!=decision:
        raise ValueError('Protocol already frozen; use new experiment for changes.')
    
    atomic_json(p,decision)
    return decision

def run_final(c,models=None):
    from .recovery import emit_capsule
    p=Path(c['output_root'])/'protocol_locked.json'
    if not p.exists():
        raise RuntimeError('Freeze protocol before accessing test.')
    
    protocol=json.loads(p.read_text())
    paths=[]
    for name in models or c['models']:
        for seed in c['seeds']:
            path=run_one(c,name,protocol['mode'],final_split(c),seed,protocol['thresholds'][name])
            paths.append(path)
            emit_capsule([path])
    return paths

def preflight(c):
    n=parameter_count(c['hidden_widths'])
    pop_bytes=c['population']*n*4
    info=dict(
        hidden_widths=c['hidden_widths'],
        trainable_parameters=n,
        one_vector_GiB=n*4/2**30,
        two_population_buffers_GiB=2*pop_bytes/2**30,
        epochs=c['epochs'],
        population=c['population'],
        other_FE=c['evaluations'],
        TAR_FE=c['tar_evaluations'],
        device=c['device'],
        cuda_available=torch.cuda.is_available(),
        notes=[
            'Disk-backed populations do not reduce candidate count or search dimensions.',
            'GPU activation/optimizer memory is additional; no runtime estimate without measuring the server.',
            'Hybrid uses 30-epoch Adam baseline then direct weight optimization; no training inside each fitness call.',
            'Population buffers are retained for resume and may consume substantial disk across folds/models.'
        ]
    )

    print(json.dumps(info,indent=2))
    return info
