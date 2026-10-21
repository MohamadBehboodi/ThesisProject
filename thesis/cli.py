import argparse,json
from pathlib import Path
from .common import config
from .data import prepare
from .pipeline import preflight,validate,select_mode,freeze_protocol,run_final
from .analysis import analyze
from .recovery import emit_capsule

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','preflight','benchmark','sequence','validate','freeze','final','analyze','all'])
    p.add_argument('--config',default='configs/default.json')
    p.add_argument('--models',nargs='+')
    p.add_argument('--output-root',help='Override the output volume/directory without changing scientific settings.')
    a=p.parse_args()
    c=config(a.config)
    if a.output_root:
        c['output_root']=str(Path(a.output_root).expanduser().resolve())

    if a.action=='benchmark':
        from .benchmark import benchmark
        report=benchmark(c)
        status=json.loads((report/'benchmark.json').read_text(encoding='utf-8'))['status']
        if status!='completed':
            raise SystemExit(2)
        return
    
    if a.action in ['preflight','all']:
        if not preflight(c)['static_checks_passed']:
            raise SystemExit('Preflight failed. Review the device and output-volume checks before running all.')
    
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
