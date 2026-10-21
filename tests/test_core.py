import contextlib,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from thesis.common import config
from thesis.data import Scaler,summary_features,batches,folds,final_split,prepare,FEATURES
from thesis.network import DLSTM,vector,assign,parameter_count
from thesis.optimizers import run_optimizer,tar_lambdas
from thesis.metrics import evaluate
from thesis.recovery import emit_capsule,restore_from_notebooks

torch.set_num_threads(2)

class CoreTests(unittest.TestCase):
    def test_padding_is_ignored(self):
        torch.manual_seed(8)
        m=DLSTM(18,[5,4,3,2]).eval()
        x=torch.randn(1,4,18)
        with torch.no_grad():
            a=m(x,torch.tensor([4]))
            b=m(torch.cat([x,torch.full((1,7,18),999.)],dim=1),torch.tensor([4]))

        self.assertTrue(torch.allclose(a,b,atol=1e-6))
        # Unsorted batch lengths must return original sample order.
        y=torch.randn(1,9,18)
        z=torch.cat([torch.nn.functional.pad(x,(0,0,0,5)),y],0)
        with torch.no_grad():
            pred=m(z,torch.tensor([4,9]))
            single=m(y,torch.tensor([9]))

        self.assertTrue(torch.allclose(pred[0],a[0],atol=1e-6))
        self.assertTrue(torch.allclose(pred[1],single[0],atol=1e-6))

    def test_vector_all_parameters(self):
        m=DLSTM(18,[5,4,3,2])
        v=vector(m)
        self.assertEqual(len(v),parameter_count([5,4,3,2]))
        v+=.125
        assign(m,v)
        np.testing.assert_array_equal(vector(m),v)

    def test_scaler_has_no_test_or_padding(self):
        xs=[np.ones((2,18)),np.ones((3,18))*3]
        s=Scaler().fit(xs)
        np.testing.assert_allclose(s.mean,2.2)
        before=s.mean.copy()
        s.transform([np.full((20,18),999.)])
        np.testing.assert_array_equal(s.mean,before)
        self.assertEqual(summary_features(xs).shape,(2,34))

    def test_rebuild_prepared_array_after_recovery(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[]
            for company in ['A','B']:
                for date in [20200101,20200102]:
                    row={f:1.0 for f in FEATURES}
                    row.update({'نماد':company,'نماد-سال':company+'-1398','سال':1398,'تاریخ میلادی':date,'سود یا زیان':int(company=='A')})
                    rows.append(row)

            pd.DataFrame(rows).to_csv(root/'data.csv',index=False)
            c=config('configs/smoke.json')
            c.update(data_path=str(root/'data.csv'),output_root=str(root/'out'),min_days=2)
            p=prepare(c)
            (p/'sequences.npz').unlink();prepare(c)
            self.assertTrue((p/'sequences.npz').exists())

    def test_optimizer_budget_and_resume(self):
        for name in ['CHOA','WCHOA','GSOBL_CHOA','TAR_CHOA','GA_reference','WOA_reference']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                c=config('configs/smoke.json')
                c.update(_seed=17,evaluations=13,tar_evaluations=23,retain_populations=True)
                calls=[]

                def objective(v):
                    calls.append(1)
                    return float(np.square(v).sum())
                
                with contextlib.redirect_stdout(io.StringIO()):
                    best,h=run_optimizer(name,objective,np.ones(11)*.2,c,Path(tmp))

                self.assertEqual(len(calls),23 if name=='TAR_CHOA' else 13)
                self.assertTrue(all(h[i]['best_fitness']<=h[i-1]['best_fitness'] for i in range(1,len(h))))

                with contextlib.redirect_stdout(io.StringIO()):
                    again,_=run_optimizer(name,objective,np.ones(11)*.2,c,Path(tmp))
                
                np.testing.assert_array_equal(best,again)
                self.assertEqual(len(calls),23 if name=='TAR_CHOA' else 13)

    def test_interrupted_generation_replay(self):
        c=config('configs/smoke.json')
        c.update(_seed=33,evaluations=16,retain_populations=True)
        def f(v):
            return float(np.square(v).sum())
        
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            with contextlib.redirect_stdout(io.StringIO()):
                expected,_=run_optimizer('CHOA',f,np.ones(11)*.2,c,Path(a))
    
            calls=0
            
            def interrupted(v):
                nonlocal calls
                calls+=1
                if calls==7:
                    raise RuntimeError('simulated process interruption')
                return f(v)
            
            with self.assertRaises(RuntimeError),contextlib.redirect_stdout(io.StringIO()):
                run_optimizer('CHOA',interrupted,np.ones(11)*.2,c,Path(b))
            
            with contextlib.redirect_stdout(io.StringIO()):
                actual,h=run_optimizer('CHOA',f,np.ones(11)*.2,c,Path(b))
            
            np.testing.assert_array_equal(actual,expected)
            self.assertGreater(len((Path(b)/'physical_evaluations.jsonl').read_text().splitlines()),16)

    def test_temporal_label_gap(self):
        c=config()
        self.assertEqual(final_split(c)['train_end'],1400)
        for s in folds(c):
            self.assertEqual(s['eval_years'][0]-s['train_end'],2)

    def test_metrics_one_class(self):
        m=evaluate([1,1],[.3,.9])
        self.assertIsNone(m['roc_auc'])
        self.assertEqual(m['fn'],1)

    def test_recovery_from_saved_output(self):
        import thesis.recovery as recovery
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'run').mkdir();(root/'run/a.json').write_text('{"metric":0.75}')
            old=recovery.ROOT
            recovery.ROOT=root
            
            try:
                s=io.StringIO()
                with contextlib.redirect_stdout(s):
                    emit_capsule([root/'run'])
            finally:
                recovery.ROOT=old
            
            (root/'test.ipynb').write_text(json.dumps({'cells':[{'outputs':[{'text':s.getvalue()}]}]}))
            with contextlib.redirect_stdout(io.StringIO()):
                restore_from_notebooks(root,root/'restore')

            self.assertEqual((root/'run/a.json').read_bytes(),(root/'restore/run/a.json').read_bytes())

    def test_tar_weights_finite(self):
        for q in [0,.1,.5,.999999]:
            a,b=tar_lambdas(q,1,.5,300000)
            self.assertAlmostEqual(a,1-q)
            self.assertAlmostEqual(b,2-2*q)

if __name__=='__main__':
    unittest.main()
