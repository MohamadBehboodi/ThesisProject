from pathlib import Path
import json, os, time, shutil
import numpy as np
from .common import atomic_json

NAMES={'CHOA','WCHOA','GSOBL_CHOA','TAR_CHOA','GA_reference','WOA_reference'}

def tar_lambdas(q,theta,r,budget,policy='literal'):
    # Printed exponent parsed as 1 - tan((r - 1/2) * pi * theta / max(phi)).
    q=np.clip(q,0,1-1e-12)
    r=np.clip(r,1e-8,1-1e-8)
    exponent=np.clip(1-np.tan((r-.5)*np.pi*theta/budget),-20,20)
    l1=float(np.exp(np.clip(exponent*np.log(max(1-q,1e-12)),-40,40)))
    l2=float(np.exp(np.clip(exponent*np.log(max(2-2*q,1e-12)),-40,40)))
    if policy=='stated_ranges': 
        return float(np.clip(l1,0,1)),float(np.clip(l2,.5,1))
    
    if policy!='literal':
        raise ValueError(policy)
    
    return l1,l2

def run_optimizer(name,fitness,initial,c,path):
    if name not in NAMES:
        raise ValueError(f'Unknown optimizer: {name}')
    
    path=Path(path)
    path.mkdir(parents=True,exist_ok=True)
    checkpoint=path/'state.json'
    n=c['population']
    d=len(initial)
    budget=c['tar_evaluations'] if name=='TAR_CHOA' else c['evaluations']

    if n<4 or budget<n:
        raise ValueError('Population >=4 and budget >= population are required.')
    
    if not checkpoint.exists() and shutil.disk_usage(path).free < 2*n*d*4+6*d*4:
        raise OSError('Insufficient disk for two full populations and working vectors. Choose a larger output volume; no automatic dimension/budget reduction.')
    
    chunk=c['vector_chunk']
    bound=c['weight_bound']
    rng=np.random.default_rng(c['_seed'])
    history=[]
    theta=np.ones(n)
    chaos=.7
    evaluations=0
    generation=0
    slot=0
    
    def mmap(s,mode): 
        return np.memmap(path/f'population_{s}.f32',dtype='float32',mode=mode,shape=(n,d))
    
    # Physical calls include work subsequently replayed after a crash. Committed FE is separate.
    ledger=path/'physical_evaluations.jsonl'
    def score(v):
        nonlocal evaluations
        if evaluations>=budget:
            return None
        
        t=time.perf_counter()
        f=float(fitness(v))
        if not np.isfinite(f): f=1e30
        
        evaluations+=1
        with ledger.open('a') as log:
            log.write(json.dumps(
                dict(
                    committed_attempt_index=evaluations,
                    fitness=f,
                    seconds=time.perf_counter()-t
                )
            )+'\n')
            log.flush()

        return f
    
    def commit(pop,scores,best,best_score):
        pop.flush()
        # Write only to the inactive slot's best file; the last committed slot stays intact.
        bpath=path/f'best_{slot}.npy'
        with open(str(bpath)+'.tmp','wb') as f:
            np.save(f,best)
        
        os.replace(str(bpath)+'.tmp',bpath)
        atomic_json(checkpoint,
            dict(
                slot=slot,
                dimension=d,
                population=n,
                name=name,
                evaluations=evaluations,
                generation=generation,
                scores=scores.tolist(),
                best_score=best_score,
                theta=theta.tolist(),
                chaos=chaos,
                rng=rng.bit_generator.state,
                history=history,
                budget=budget
            )
        )

    if checkpoint.exists():
        s=json.loads(checkpoint.read_text())
        if (s['name'],s['dimension'],s['population'],s['budget'])!=(name,d,n,budget):
            raise ValueError('Optimizer resume configuration mismatch.')
        
        slot=s['slot']
        evaluations=s['evaluations']
        generation=s['generation']
        scores=np.array(s['scores'])
        theta=np.array(s['theta'])
        chaos=s['chaos']
        history=s['history']
        rng.bit_generator.state=s['rng']
        pop=mmap(slot,'r+')
        best=np.load(path/f'best_{slot}.npy')
        best_score=s['best_score']

    else:
        pop=mmap(0,'w+')
        for i in range(n):
            for j in range(0,d,chunk):
                k=min(j+chunk,d)
                
                if i==0: z=initial[j:k]
                elif c['population_initialization']=='warm_normal': z=initial[j:k]+rng.normal(0,c['initial_std'],k-j)
                elif c['population_initialization']=='uniform': z=rng.uniform(-bound,bound,k-j)
                else: raise ValueError('Unknown population initialization.')
                
                pop[i,j:k]=np.clip(z,-bound,bound)

        scores=np.array([score(pop[i]) for i in range(n)])
        b=int(np.argmin(scores))
        best=np.array(pop[b])
        best_score=float(scores[b])
        history.append(
            dict(
                generation=0,
                evaluations=evaluations,
                best_fitness=best_score
            )
        )
        commit(pop,scores,best,best_score)

    while evaluations<budget:
        old=pop
        old_scores=scores.copy()
        slot=1-slot
        pop=mmap(slot,'w+')
        scores=old_scores.copy()

        # The archived best joins current candidates; four unique candidates by index.
        leader_ids=np.argsort(old_scores)[:4]
        leaders=[np.array(old[i]) for i in leader_ids]
        if best_score<old_scores[leader_ids[0]]:
            leaders=[best.copy()]+leaders[:3]

        q=evaluations/budget
        f=2*(1-q)
        chaos=0.0 if chaos==0 else (1/chaos)%1
        
        # Prevent a finite-precision absorbing state; deterministic documented restart.
        if chaos<1e-12: chaos=.7
        
        for i in range(n):
            pop[i]=old[i]
            if evaluations>=budget: 
                continue

            x=np.array(old[i])
            candidate=np.empty(d,dtype=np.float32)
            if name=='GA_reference':
                parents=[]
                for _ in range(2):
                    ids=rng.choice(n,size=3,replace=False)
                    #parents.append(old[int(ids[np.argmin(old_scores[ids])])]) -- Original code
                    parents.append(old[int(ids[np.argmin(old_scores[ids])])].copy())

                for j in range(0,d,chunk):
                    k=min(d,j+chunk)
                    a=rng.random(k-j)
                    z=a*parents[0][j:k]+(1-a)*parents[1][j:k]
                    mutation=rng.random(k-j)<c['ga_mutation_probability']
                    candidate[j:k]=np.clip(z+mutation*rng.normal(0,c['ga_mutation_std'],k-j),-bound,bound)

            elif name=='WOA_reference':
                a=2*(1-q)
                A=2*a*rng.random()-a
                C=2*rng.random()
                p=rng.random()
                ell=rng.uniform(-1,1)
                #target=best if abs(A)<1 else old[int(rng.integers(n))] -- Original Code
                target=np.array(best if abs(A)<1 else old[int(rng.integers(n))], copy=True)

                for j in range(0,d,chunk):
                    k=min(d,j+chunk)
                    z=target[j:k]-A*np.abs(C*target[j:k]-x[j:k]) if p<.5 else np.abs(best[j:k]-x[j:k])*np.exp(ell)*np.cos(2*np.pi*ell)+best[j:k]
                    candidate[j:k]=np.clip(z,-bound,bound)

            else:
                # corOppose structure from official GSOBL source: first row reference,
                # coordinate-distance statistic, threshold f, indices >= 4 (MATLAB).
                # Fix size(vector) ambiguity with scalar counts; add explicit greedy acceptance.
                gsobl_incumbent=old_scores[i]
                if name=='GSOBL_CHOA' and i>=3:
                    distances=np.abs(old[0]-x)
                    selected=distances<f
                    src=1-6*float(np.square(distances.astype(np.float64)).sum())/(n*(n*n-1))
                    if src<=0 and int(selected.sum())>=int((~selected).sum()):
                        opposite=x.copy()
                        opposite[selected]=-opposite[selected]
                        fo=score(opposite)
                        if fo is not None and fo<gsobl_incumbent:
                            x=opposite
                            gsobl_incumbent=fo
                        
                        if fo is not None and fo<best_score:
                            best_score=float(fo)
                            best=opposite.copy()

                def chimp(lam=1.0):
                    z=np.empty(d,dtype=np.float32)
                    for j in range(0,d,chunk):
                        k=min(d,j+chunk); terms=[]
                        
                        for leader in leaders:
                            A=2*f*rng.random(k-j)-f
                            C=2*rng.random(k-j)
                            terms.append(lam*leader[j:k]-A*np.abs(C*leader[j:k]-chaos*x[j:k]))

                        t=np.stack(terms)
                        if name in {'WCHOA','GSOBL_CHOA'}:
                            abs_t=np.abs(t)
                            den=abs_t.sum(0)
                            weights=np.divide(abs_t,den,out=np.full_like(abs_t,.25),where=den>1e-12)
                            result=(weights*t).sum(0) # corrected normalized weighted mean, no extra /4
                        else:
                            result=t.mean(0)
                        
                        z[j:k]=np.clip(result,-bound,bound)
                    return z
                
                if name=='TAR_CHOA':
                    # Fig 6 stochastic alternative; unspecified replacement distribution -> uniform bounds.
                    alternative=x.copy()
                    for j in range(0,d,chunk):
                        k=min(d,j+chunk)
                        r=np.clip(rng.random(k-j),1e-8,1-1e-8)
                        mask=np.tan(np.pi*(r-.5))<1-q
                        alternative[j:k]=np.where(mask,rng.uniform(-bound,bound,k-j),alternative[j:k])

                    alt_score=score(alternative)
                    if alt_score is not None and alt_score<old_scores[i]:
                        x=alternative
                        current_score=alt_score
                    else:
                        current_score=old_scores[i]
                    
                    options=[(current_score,x.copy())]
                    l1,l2=tar_lambdas(evaluations/budget,theta[i],rng.random(),budget,c['tar_lambda_policy'])
                    base=chimp()
                    fb=score(base)
                    
                    if fb is not None: options.append((fb,base))
                    
                    # Evaluate both candidates as depicted; only active-half candidate enters acceptance.
                    for lam,active in [(l1,q<.5),(l2,q>=.5)]:
                        if evaluations>=budget:
                            break
                        
                        z=chimp(lam)
                        fz=score(z)
                        
                        if active: options.append((fz,z))
                        if fz<best_score:
                            best_score=float(fz)
                            best=z.copy()

                    chosen=min(options,key=lambda pair:pair[0])
                    candidate=chosen[1]
                    val=float(chosen[0])
                    theta[i]=theta[i]/2 if val<old_scores[i] else theta[i]+1

                else: 
                    candidate=chimp()

            if name!='TAR_CHOA':
                val=score(candidate)
                if name=='GSOBL_CHOA' and (val is None or gsobl_incumbent<val):
                    candidate=x
                    val=float(gsobl_incumbent)
                
                if val is None:
                    continue

            if val<best_score:
                best_score=float(val)
                best=candidate.copy()

            # Source CHOA/WOA move all agents. Greedy variants retain better incumbent.
            greedy=name in {'GSOBL_CHOA','TAR_CHOA','GA_reference'}
            if not greedy or val<old_scores[i]:
                pop[i]=candidate
                scores[i]=val
        
        generation+=1
        history.append(
            dict(
                generation=generation,
                evaluations=evaluations,
                best_fitness=best_score
            )
        )
        
        commit(pop,scores,best,best_score)
        atomic_json(path/'convergence.json',history)
        print(f'{name} generation={generation} FE={evaluations}/{budget} fitness={best_score:.7g}',flush=True)
        del old,leaders
    return best,history
