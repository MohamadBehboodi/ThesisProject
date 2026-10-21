from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import atomic_json
from .metrics import evaluate


def fitness_costs(ledger):
    calls, seconds = 0, 0.0
    if ledger.exists():
        with ledger.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                calls += 1
                seconds += row['seconds']
    return calls, seconds

def analyze(c):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay,PrecisionRecallDisplay,ConfusionMatrixDisplay
    
    root=Path(c['output_root'])
    from .pipeline import check_experiment
    p=json.loads((root/'protocol_locked.json').read_text(encoding='utf-8'))
    if p['key'] != check_experiment(c):
        raise ValueError('Analysis config differs from the frozen experiment.')
    mode=p['mode']
    out=root/'analysis'
    out.mkdir(exist_ok=True)
    rows=[]
    ensembles={}
    seed_rows=[]
    cost_rows=[]
    shared_warm = {}

    for name in c['models']:
        parts=[]
        for seed in c['seeds']:
            run=root/'runs'/mode/'test'/name/f'seed_{seed}'/'test'
            if not (run/'complete.json').exists():
                raise FileNotFoundError(f'Incomplete final run: {run}')
            
            f=pd.read_csv(run/'predictions.csv') 
            parts.append(f)
            seed_rows.append(
                dict(
                    model=name,
                    seed=seed,
                    **evaluate(f.y,f.probability,p['thresholds'][name])
                )
            )
            
            complete=json.loads((run/'complete.json').read_text(encoding='utf-8'))
            ledger=run/'optimizer'/'physical_evaluations.jsonl'
            physical_fe, fitness_seconds = fitness_costs(ledger)
            timing_path = run/'optimizer'/'timing.json'
            optimizer_timing = json.loads(timing_path.read_text(encoding='utf-8')) if timing_path.exists() else {}
            diagnostics_path = run/'optimizer'/'diagnostics.json'
            diagnostics = json.loads(diagnostics_path.read_text(encoding='utf-8')) if diagnostics_path.exists() else {}
            shared = complete.get('shared_baseline_path')
            if shared and (Path(shared)/'training_timing.json').exists():
                shared_warm[shared] = json.loads((Path(shared)/'training_timing.json').read_text(encoding='utf-8'))
            
            cost_rows.append(
                dict(
                    model=name,
                    seed=seed,
                    run_seconds=complete['seconds'],
                    physical_FE=physical_fe,
                    committed_FE=diagnostics.get('committed_evaluations', 0),
                    fitness_seconds=fitness_seconds,
                    optimizer_active_seconds=optimizer_timing.get('seconds', 0),
                    timing_uncertain_sessions=complete.get('timing_uncertain_sessions', 0),
                    baseline_access_seconds_last_invocation=complete.get('baseline_access_seconds_last_invocation', 0),
                    training_fitness_improvement=diagnostics.get('training_fitness_improvement'),
                    changed_parameters=diagnostics.get('vector_change', {}).get('changed_parameters'),
                    note='Run time accumulates active invocations; shared warm training is reported once separately.'
                )
            )

            history=json.loads((run/'history.json').read_text(encoding='utf-8'))
            if history and 'evaluations' in history[0]:
                fig,ax=plt.subplots(figsize=(6,4))
                ax.plot([x['evaluations'] for x in history],[x['best_fitness'] for x in history])
                ax.set(xlabel='Committed fitness evaluations',ylabel='Best training fitness',title=f'{name}, seed {seed}')
                fig.tight_layout()
                fig.savefig(out/f'{name}_seed_{seed}_convergence.png',dpi=160)
                plt.close(fig)
        
        allf=pd.concat(parts)
        if allf.groupby('sample_id').y.nunique().max()!=1:
            raise ValueError('Label conflict.')
        
        ens=allf.groupby('sample_id',sort=True).agg(
            symbol=('symbol','first'),
            year=('year','first'),
            y=('y','first'),
            probability=('probability','mean'),
            seeds=('seed','nunique')
        ).reset_index()

        if not (ens.seeds==len(c['seeds'])).all():
            raise ValueError('Incomplete ensemble.')
        
        ens['prediction']=(ens.probability>=p['thresholds'][name]).astype(int)
        ens.to_csv(out/f'{name}_ensemble_predictions.csv',index=False)
        ensembles[name]=ens
        rows.append(
            dict(
                model=name,
                **evaluate(ens.y,ens.probability,p['thresholds'][name])
            )
        )

        fig,ax=plt.subplots(1,4,figsize=(16,3.5))
        if ens.y.nunique()==2:
            RocCurveDisplay.from_predictions(ens.y,ens.probability,ax=ax[0])
            PrecisionRecallDisplay.from_predictions(ens.y,ens.probability,ax=ax[1])
            PrecisionRecallDisplay.from_predictions(1-ens.y,1-ens.probability,ax=ax[3])
        ax[0].set_title('ROC: profit positive')
        ax[1].set_title('PR: profit positive')
        ax[3].set_title('PR: loss positive')

        ConfusionMatrixDisplay.from_predictions(ens.y,ens.prediction,labels=[0,1],
                                               display_labels=['Loss (0)', 'Profit (1)'],ax=ax[2])
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(out/f'{name}_evaluation.png',dpi=160)
        plt.close(fig)
        
        for year,g in ens.groupby('year'):
            atomic_json(out/f'{name}_year_{year}.json',evaluate(g.y,g.probability,p['thresholds'][name]))

    pd.DataFrame(rows).to_csv(out/'model_comparison.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(out/'seed_metrics.csv',index=False)
    pd.DataFrame(cost_rows).to_csv(out/'compute_costs.csv',index=False)
    atomic_json(out/'shared_warm_training_costs.json', dict(
        checkpoints=shared_warm, total_seconds=sum(s['seconds'] for s in shared_warm.values()),
        note='Unique final-split warm checkpoints; do not charge this total separately to every hybrid.'
    ))
    from .resources import experiment_resources
    atomic_json(out/'experiment_budget.json', experiment_resources(c))

    comparisons=[]
    base=ensembles['DLSTM'].set_index('sample_id').sort_index()
    for name,frame in ensembles.items():
        if name=='DLSTM':
            continue
        
        other=frame.set_index('sample_id').sort_index()
        if not base.index.equals(other.index) or not np.array_equal(base.y,other.y):
            raise ValueError('Unpaired test sample sets.')
        
        symbols=base.symbol.unique()
        rng=np.random.default_rng(314159)
        groups={s:np.flatnonzero(base.symbol.to_numpy()==s) for s in symbols}
        diffs=[]

        def ba(y,pred):
            return .5*((pred[y==1]==1).mean()+(pred[y==0]==0).mean())
        
        y=base.y.to_numpy()
        a=base.prediction.to_numpy()
        b=other.prediction.to_numpy()

        if len(np.unique(y))<2:
            raise ValueError('Final test requires both classes for paired Balanced Accuracy.')
        
        observed=float(ba(y,b)-ba(y,a))
        for _ in range(c['bootstrap_repeats']):
            ids=np.concatenate([groups[s] for s in rng.choice(symbols,size=len(symbols),replace=True)])
            if len(np.unique(y[ids]))<2:
                continue
            
            diffs.append(float(ba(y[ids],b[ids])-ba(y[ids],a[ids])))

        exceed=0
        for _ in range(c['bootstrap_repeats']):
            swap=np.zeros(len(y),dtype=bool)
            for s,flip in zip(symbols,rng.integers(0,2,len(symbols))):
                swap[groups[s]]=bool(flip)
            
            aa=np.where(swap,b,a)
            bb=np.where(swap,a,b)
            exceed+=abs(ba(y,bb)-ba(y,aa))>=abs(observed)-1e-12
        
        comparisons.append(
            dict(
                model=name,
                reference='DLSTM',
                delta_balanced_accuracy=observed,
                ci_low=float(np.quantile(diffs,.025)) if diffs else None,
                ci_high=float(np.quantile(diffs,.975)) if diffs else None,
                cluster_randomization_p=(1+exceed)/(1+c['bootstrap_repeats']),
                valid_bootstrap=len(diffs),
                clusters=len(symbols)
            )
        )

    
    ordered=sorted(range(len(comparisons)),key=lambda i:comparisons[i]['cluster_randomization_p'])
    previous=0
    for rank,i in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-rank)*comparisons[i]['cluster_randomization_p']))
        comparisons[i]['holm_p']=previous
    
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False)
    atomic_json(out/'analysis_notes.json',
        dict(
            primary='seed-ensemble Balanced Accuracy',
            uncertainty='company-cluster bootstrap',
            testing='paired company-cluster model-swap randomization; Holm-adjusted',
            limitations=[
                'Only the available test years; no claim of market-wide temporal generalization.',
                'Prior versions of the thesis already examined these test years; this is not a newly unseen external test.',
                'Seeds are computational repetitions, not additional independent financial observations.'
            ]
        )
    )
    return out
