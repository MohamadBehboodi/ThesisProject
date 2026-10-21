import argparse,json
from .common import config
from .data import prepare
from .pipeline import preflight,validate,select_mode,freeze_protocol,run_final
from .analysis import analyze
from .recovery import emit_capsule

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','preflight','sequence','validate','freeze','final','analyze','all'])
    p.add_argument('--config',default='configs/default.json')
    p.add_argument('--models',nargs='+')
    a=p.parse_args()
    c=config(a.config)
    
    if a.action in ['preflight','all']:
        preflight(c)
    
    if a.action in ['prepare','all']:
        print(prepare(c))
    
    if a.action in ['sequence','all']:
        validate(c,['DLSTM'],c['sequence_modes'])
        print(select_mode(c))
    
    if a.action in ['validate','all']:
        validate(c,a.models or c['models'])
    
    if a.action in ['freeze','all']:
        print(freeze_protocol(c))
    
    if a.action in ['final','all']:
        run_final(c,a.models)
    
    if a.action in ['analyze','all']:
        print(analyze(c))
    
    if a.action!='preflight':
        emit_capsule([c['output_root']])

if __name__=='__main__':
    main()
