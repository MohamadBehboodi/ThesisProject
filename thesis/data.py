from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_json, digest, signature

FEATURES = ['بازده فروش','بازده دارایی','بازده سرمایه','نسبت جاری','نسبت آنی','نسبت نقدینگی',
 'نسبت بدهی','نسبت بدهی به سرمایه','نسبت پوشش بدهی','نسبت دارایی های جاری',
 'نسبت قیمت به درآمد','نسبت قیمت به فروش','بیشترین قیمت','کمترین قیمت',
 'قیمت آخرین معامله','قیمت باز شدن','ارزش معاملات','حجم معاملات']

def prepare(c):
    dest=Path(c['output_root'])/'prepared'
    dest.mkdir(parents=True,exist_ok=True)
    fp=digest(c['data_path'])
    key=signature(
        dict(
            input_sha256=fp,
            min_days=c['min_days'],
            features=FEATURES
        )
    )

    if (dest/'metadata.json').exists():
        old=__import__('json').loads((dest/'metadata.json').read_text(encoding='utf-8')) # Original code doesn't have encoding='utf-8'
        if old['fingerprint']!=key:
            raise ValueError('Prepared data differ. Use a new output_root; never overwrite an experiment.')
        
        if all((dest/name).exists() for name in ['sequences.npz','samples.csv','excluded.csv']):
            return dest

    df=pd.read_csv(c[ 'data_path'])
    required=FEATURES+['نماد','نماد-سال','سال','تاریخ میلادی','سود یا زیان']
    if set(required)-set(df):
        raise ValueError(f'Missing columns: {set(required)-set(df)}')
    
    df['date']=pd.to_datetime(df['تاریخ میلادی'].astype(str),format='%Y%m%d',errors='raise')
    if df[required].isna().any().any(): 
        raise ValueError('Missing raw values: resolve explicitly before training.')
    
    if df.duplicated(['نماد','date']).any(): 
        raise ValueError('Duplicate company/date rows.')
    
    a=df[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(a).all(): 
        raise ValueError('Nonfinite raw features.')
    
    if not set(df['سود یا زیان'].unique()) <= {0,1}:
        raise ValueError('Target must be 0/1.')

    a[:,10]=np.sign(a[:,10])*np.log1p(np.abs(a[:,10]))
    if (a[:,11:]<0).any():
        raise ValueError('Negative value in a nonnegative log1p feature.')
    
    a[:,11:]=np.log1p(a[:,11:])
    df[FEATURES]=a
    rows=[]; sequences=[]; excluded=[]; offsets=[0]
    for key_id,g in df.groupby('نماد-سال',sort=True):
        g=g.sort_values('date',kind='stable')
        if any(g[k].nunique()!=1 for k in ['سال','نماد','سود یا زیان']):
            raise ValueError(f'Inconsistent identity/year/label: {key_id}')
        
        if len(g)<c['min_days']:
            excluded.append({'sample_id':key_id,'n_days':len(g)})
            continue
        
        sequences.append(g[FEATURES].to_numpy(dtype=np.float32))
        offsets.append(offsets[-1]+len(g))
        rows.append(
            dict(
                sample_id=key_id,
                symbol=g['نماد'].iloc[0],
                year=int(g['سال'].iloc[0]),
                y=int(g['سود یا زیان'].iloc[0]),
                length=len(g),
                first_date=str(g['date'].iloc[0].date()),
                last_date=str(g['date'].iloc[-1].date())
            )
        )
    
    np.savez_compressed(dest/'sequences.npz',values=np.concatenate(sequences),offsets=np.array(offsets))
    pd.DataFrame(rows).to_csv(dest/'samples.csv',index=False)
    pd.DataFrame(excluded,columns=['sample_id','n_days']).to_csv(dest/'excluded.csv',index=False)
    
    atomic_json(dest/'metadata.json',
        dict(
            fingerprint=key,
            input_sha256=fp,
            raw_rows=len(df),
            samples=len(rows),
            retained_rows=offsets[-1],
            excluded_samples=len(excluded),
            features=FEATURES,
            min_days=c['min_days'],
            max_days=max(x['length'] for x in rows),
            label='Already supplied: company-year t -> profitability t+1; no extra shift.',
            information_availability='Forecast-time availability is verified.'
        )
    )

    return dest

def load_data(c,mode):
    dest=prepare(c)
    meta=pd.read_csv(dest/'samples.csv')
    with np.load(dest/'sequences.npz') as z:
        values=z['values'];offsets=z['offsets']

    xs=[values[l:r].copy() for l,r in zip(offsets[:-1],offsets[1:])]
    if mode=='last100':
        xs=[x[-100:] for x in xs]
    elif mode!='full_year':
        raise ValueError(mode)
    
    return xs,meta

def folds(c):
    return [
        dict(
            name=f'val_{v}',
            train_end=v-1-c['label_gap_years'],
            eval_years=[v],
            stage='validation'
        )
        for v in c['validation_years']
    ]

def final_split(c):
    return dict(
        name='test',
        train_end=min(c['test_years'])-1-c['label_gap_years'],
        eval_years=c['test_years'],
        stage='test'
    )

def indices(meta,c,split):
    tr=np.flatnonzero(meta.year.between(c['start_year'],split['train_end']).to_numpy())
    ev=np.flatnonzero(meta.year.isin(split['eval_years']).to_numpy())
    if not len(tr) or not len(ev):
        raise ValueError('Empty temporal split.')
    
    if len(np.unique(meta.y.iloc[tr]))<2:
        raise ValueError('Training set contains one class.')
    
    assert meta.year.iloc[tr].max()<meta.year.iloc[ev].min()
    return tr,ev

class Scaler:
    def fit(self,xs):
        x=np.concatenate(xs).astype(np.float64)
        self.mean=x.mean(0)
        self.scale=x.std(0)
        self.scale[self.scale<1e-12]=1
        return self
    
    def transform(self,xs):
        return [((x-self.mean)/self.scale).astype(np.float32) for x in xs]
    
    def state(self):
        return dict(
            mean=self.mean.tolist(),
            scale=self.scale.tolist(),
            features=FEATURES
        )

def summary_features(xs):
    return np.array([np.concatenate([x[-1,:10],x[-1,10:],x[:,10:].mean(0),x[:,10:].std(0)]) for x in xs])

def batches(xs,ys,batch_size,shuffle=False):
    import torch
    ids=np.arange(len(xs))
    if shuffle:
        np.random.shuffle(ids)

    for start in range(0,len(ids),batch_size):
        ii=ids[start:start+batch_size]
        lengths=np.array([len(xs[i]) for i in ii])
        a=np.zeros((len(ii),int(lengths.max()),xs[0].shape[1]),dtype=np.float32)
        for k,i in enumerate(ii):
            a[k,:lengths[k]]=xs[i]

        yield torch.from_numpy(a),torch.from_numpy(lengths),torch.tensor(np.asarray(ys)[ii],dtype=torch.long)
