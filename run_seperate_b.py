"""Train known-deformation specialists, validate, then evaluate and plot.

All amplitudes are b/lambda, not displacement in metres.
python run_separate_b.py --output b_sweep
python run_separate_b.py --output b_sweep --resume
"""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('separate_b_results'))
    p.add_argument('--alphas',nargs='+',type=float,default=[.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
    p.add_argument('--seeds',nargs='+',type=int,default=[11])
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--validation-scenes',type=int,default=256)
    p.add_argument('--test-scenes',type=int,default=500)
    p.add_argument('--snrs',nargs='+',type=float,default=[-5,0,5,10,15,20,25,30])
    p.add_argument('--include-direct',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--quick',action='store_true',help='Tiny software test at all requested amplitudes')
    a=p.parse_args()
    import math
    if not a.alphas or any(not math.isfinite(x) or x<=0 for x in a.alphas) or len(set(a.alphas))!=len(a.alphas):p.error('Unique positive finite b/lambda values required')
    if not {0,10,20}.issubset(a.snrs):p.error('Include 0,10,20 in --snrs for the fixed-SNR plots')
    if a.quick:a.seeds=[11];a.test_scenes=8
    a.output=a.output.resolve();project=Path(__file__).resolve().parent
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('output','resume')}
    if a.output.exists() and any(a.output.iterdir()):
        if not a.resume:p.error('Use a new output folder or --resume')
        if json.loads((a.output/'sweep_configuration.json').read_text())!=config:p.error('Resume requires identical settings')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'sweep_configuration.json').write_text(json.dumps(config,indent=2))
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    subprocess.run([sys.executable,str(project/'train.py'),'--help'],cwd=project,env=env,check=True,stdout=subprocess.DEVNULL)
    import inspect
    import train
    from evaluate_sweep import render
    if '--regimes' not in inspect.getsource(train.main) or '--model-kinds' not in inspect.getsource(train.main):
        raise RuntimeError('Replace train.py with the supplied matching version before running this sweep')
    runs=[]
    for index,alpha in enumerate(a.alphas):
        folder=a.output/f'b_{alpha:.12g}';runs.append(folder)
        print(f'\n=== Deformation {index+1}/{len(a.alphas)}: b/lambda={alpha:g} ===',flush=True)
        command=[sys.executable,str(project/'train.py'),'--train-only','--regimes','large',
            '--model-kinds','hybrid',*(['direct'] if a.include_direct else []),
            '--large-alpha',str(alpha),'--small-alpha','0','--output',str(folder),
            '--epochs',str(a.epochs),'--steps',str(a.steps),'--batch-size',str(a.batch_size),
            '--validation-scenes',str(a.validation_scenes),'--seeds',*map(str,a.seeds)]
        if a.quick:command+=['--quick']
        if a.resume and (folder/'configuration.json').exists():command+=['--resume']
        subprocess.run(command,cwd=project,env=env,check=True)
    evaluation=a.output/'evaluation'
    if not (a.resume and (evaluation/'run_status.json').exists() and json.loads((evaluation/'run_status.json').read_text()).get('completed')):
        # A previous interrupted evaluation is preserved; rerun in a new folder.
        if evaluation.exists() and any(evaluation.iterdir()):
            import tempfile
            evaluation=Path(tempfile.mkdtemp(prefix='evaluation_retry_',dir=a.output))
        command=[sys.executable,str(project/'evaluate_sweep.py'),'--results',*map(str,runs),
                 '--output',str(evaluation),'--scenes',str(a.test_scenes),'--snrs',*map(str,a.snrs)]
        if a.include_direct:command+=['--include-direct']
        if a.quick:command+=['--allow-smoke']
        subprocess.run(command,cwd=project,env=env,check=True)
    # Add validation-versus-epoch plots; final SNR plots are test-set metrics.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(16,6));ax=axes[1];best=[]
    for alpha,folder in zip(a.alphas,runs):
        for seed in a.seeds:
            for kind in ['hybrid']+(['direct'] if a.include_direct else []):
                rows=list(csv.DictReader((folder/f'seed_{seed}'/f'large_{kind}'/'history.csv').open()))
                if kind=='hybrid':
                    axes[0].plot([int(r['epoch']) for r in rows],10*np.log10([float(r['train_nmse']) for r in rows]),label=f'b/λ={alpha:g}, seed={seed}')
                    ax.plot([int(r['epoch']) for r in rows],10*np.log10([float(r['validation_nmse']) for r in rows]),label=f'b/λ={alpha:g}, seed={seed}')
                initial=json.loads((folder/f'seed_{seed}'/f'large_{kind}'/'initial_validation.json').read_text())['nmse']
                r=min(rows,key=lambda r:float(r['validation_nmse']))
                val=min(initial,float(r['validation_nmse']))
                best.append(dict(alpha=alpha,seed=seed,kind=kind,best_epoch=0 if initial<=float(r['validation_nmse']) else int(r['epoch']),validation_nmse=val,validation_nmse_db=10*math.log10(val)))
    ax.set(xlabel='Epoch',ylabel='Validation NMSE (dB)',title='Known-deformation hybrid specialists — validation history')
    axes[0].set(xlabel='Epoch',ylabel='Training NMSE (dB)',title='Independent specialists — training history')
    for axis in axes:
        axis.grid(alpha=.25);axis.legend(fontsize=8,ncol=2)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(evaluation/f'04_training_validation_history.{ext}',dpi=180)
    plt.close(fig)
    with (evaluation/'best_validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(best[0]));w.writeheader();w.writerows(best)
    (a.output/'sweep_status.json').write_text(json.dumps({'completed':True,'evaluation_folder':str(evaluation)}))
    print('Sweep complete. Plots and test results:',evaluation,flush=True)


if __name__=='__main__':main()
