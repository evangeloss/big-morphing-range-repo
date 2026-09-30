"""Evaluate known-deformation checkpoints and plot three comparisons.

Add beside simulator.py, bridge.py and model.py. No training files need editing.
Example:
  python evaluate_sweep.py --results results_b05 results_b1 --output sweep
Requires FULL saved results directories, including .pt files, not review ZIPs.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import Simulator
from bridge import PhysicsBridge
from model import Reconstructor,nmse


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def same_state(left,right):
    return left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def render(rows,gains,out,fixed_snrs):
    methods=list(dict.fromkeys(r['method'] for r in rows))
    names={'hybrid':'Physics + learned correction','physics':'Physics only','direct':'Direct prediction'}
    alphas=sorted({r['alpha'] for r in rows})
    colors=plt.get_cmap('viridis')(np.linspace(.08,.88,len(alphas)))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,name):
        fig.tight_layout(rect=(0,.04,1,.93))
        fig.text(.5,.015,'Mean linear NMSE is averaged across scenes and selected training seeds before dB conversion.',ha='center',fontsize=8)
        for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=190,bbox_inches='tight')
        plt.close(fig)
    for scale in ['linear','db']:
        key='mean_nmse' if scale=='linear' else 'nmse_db'
        ylabel='Mean NMSE (linear)' if scale=='linear' else 'NMSE (dB)'
        fig,axs=plt.subplots(1,len(methods),figsize=(6*len(methods),5.5),squeeze=False)
        for ax,method in zip(axs[0],methods):
            for alpha,color in zip(alphas,colors):
                rs=sorted([r for r in rows if r['method']==method and r['alpha']==alpha],key=lambda r:r['snr_db'])
                ax.plot([r['snr_db'] for r in rs],[r[key] for r in rs],'o-',color=color,label=f'b/λ={alpha:g}')
            ax.set(title=names[method],xlabel='SNR (dB)',ylabel=ylabel);ax.grid(alpha=.25);ax.legend(fontsize=8)
            if scale=='linear':ax.set_ylim(bottom=0)
        fig.suptitle('1. NMSE versus SNR — known deformation',fontsize=15)
        save(fig,f'01_nmse_vs_snr_{scale}')
        fig,axs=plt.subplots(1,len(methods),figsize=(6*len(methods),5.5),squeeze=False)
        for ax,method in zip(axs[0],methods):
            for snr in fixed_snrs:
                rs=sorted([r for r in rows if r['method']==method and r['snr_db']==snr],key=lambda r:r['alpha'])
                ax.plot([r['alpha'] for r in rs],[r[key] for r in rs],'o-',label=f'SNR={snr:g} dB')
            ax.set(title=names[method],xlabel='Deformation b/λ',ylabel=ylabel);ax.grid(alpha=.25);ax.legend(fontsize=8)
            if scale=='linear':ax.set_ylim(bottom=0)
        fig.suptitle('2. NMSE versus deformation',fontsize=15)
        save(fig,f'02_nmse_vs_deformation_{scale}')
    fig,axs=plt.subplots(1,2,figsize=(12,5.5))
    for alpha,color in zip(alphas,colors):
        rs=sorted([r for r in gains if r['alpha']==alpha],key=lambda r:r['snr_db'])
        axs[0].plot([r['snr_db'] for r in rs],[r['gain_db'] for r in rs],'o-',color=color,label=f'b/λ={alpha:g}')
    for snr in fixed_snrs:
        rs=sorted([r for r in gains if r['snr_db']==snr],key=lambda r:r['alpha'])
        axs[1].plot([r['alpha'] for r in rs],[r['gain_db'] for r in rs],'o-',label=f'SNR={snr:g} dB')
    for ax,xlabel in zip(axs,['SNR (dB)','Deformation b/λ']):
        ax.axhline(0,color='black',linewidth=1,linestyle='--');ax.set(xlabel=xlabel,ylabel='Physics NMSE (dB) − hybrid NMSE (dB)')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('3. Gain from learned correction — positive is better',fontsize=15)
    save(fig,'03_hybrid_gain')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('evaluation_sweep'))
    p.add_argument('--snrs',nargs='+',type=float,default=[-5,0,5,10,15,20,25,30])
    p.add_argument('--fixed-snrs',nargs='+',type=float,default=[0,10,20])
    p.add_argument('--alphas',nargs='+',type=float,help='Optional subset of available trained amplitudes')
    p.add_argument('--seeds',nargs='+',type=int,help='Default: common training seeds across all supplied runs')
    p.add_argument('--scenes',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--evaluation-seed',type=int,default=3000000000)
    p.add_argument('--include-direct',action='store_true')
    p.add_argument('--allow-smoke',action='store_true',help='Allow checkpoints from --quick runs for software testing only')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args()
    if a.scenes<1 or a.batch_size<1 or a.evaluation_seed<0:p.error('Positive scene/batch counts and nonnegative evaluation seed required')
    if not all(math.isfinite(x) for x in a.snrs+a.fixed_snrs):p.error('SNRs must be finite')
    a.snrs=sorted(set(a.snrs));a.fixed_snrs=sorted(set(a.fixed_snrs))
    if not set(a.fixed_snrs).issubset(a.snrs):p.error('--fixed-snrs must be included in --snrs')
    if a.output.exists() and any(a.output.iterdir()):p.error('Use a fresh output directory')
    configs=[]
    for root in a.results:
        f=root/'configuration.json'
        if not f.is_file():p.error(f'Missing {f}. Supply full results directories, not review ZIPs.')
        cfg=json.loads(f.read_text());configs.append((root,cfg))
        if cfg.get('quick') and not a.allow_smoke:p.error('Smoke-test checkpoints cannot be used for scientific evaluation')
    reference=configs[0][1]
    for root,cfg in configs:
        for key in ['paths','geometry_seed','pair_start']:
            if cfg[key]!=reference[key]:p.error(f'{key} differs across runs; use separate sweeps')
    seeds=sorted(set(a.seeds)) if a.seeds else sorted(set.intersection(*(set(c['seeds']) for _,c in configs)))
    if not seeds:p.error('No common training seeds. Pass compatible run directories.')
    if any(not set(seeds).issubset(c['seeds']) for _,c in configs):p.error('Every run must contain all selected seeds')
    sources={};duplicates=[]
    for root,cfg in configs:
        for regime in ['small','large']:
            alpha=float(cfg[f'{regime}_alpha'])
            if a.alphas is not None and not any(abs(alpha-x)<1e-9 for x in a.alphas):continue
            candidate=(root,cfg,regime)
            if alpha in sources:duplicates.append((alpha,sources[alpha],candidate))
            else:sources[alpha]=candidate
    if not sources:p.error('No requested trained amplitudes found')
    if a.alphas and any(not any(abs(x-y)<1e-9 for y in sources) for x in a.alphas):p.error('A requested amplitude has no trained specialist; train it first')
    kinds=['hybrid']+(['direct'] if a.include_direct else [])
    # Compare tensor contents, not torch archive bytes, to deduplicate repeated small references.
    for alpha,left,right in duplicates:
        lb=torch.load(left[0]/f'bridge_{left[2]}.pt',map_location='cpu',weights_only=True)
        rb=torch.load(right[0]/f'bridge_{right[2]}.pt',map_location='cpu',weights_only=True)
        if not same_state({k:lb[k] for k in ['U','T','eigenvalues']},{k:rb[k] for k in ['U','T','eigenvalues']}):
            p.error(f'Different bridges for duplicate alpha={alpha}. Supply only one source for that amplitude.')
        for seed in seeds:
            for kind in kinds:
                ls=torch.load(left[0]/f'seed_{seed}'/f'{left[2]}_{kind}'/'best.pt',map_location='cpu',weights_only=True)['model']
                rs=torch.load(right[0]/f'seed_{seed}'/f'{right[2]}_{kind}'/'best.pt',map_location='cpu',weights_only=True)['model']
                if not same_state(ls,rs):p.error(f'Different checkpoints for duplicate alpha={alpha}; supply one source or use --alphas to select a subset')
        print(f'Deduplicated identical reference at alpha={alpha:g}',flush=True)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    sim=Simulator(reference['geometry_seed'],reference['paths'],reference['pair_start']).to(device)
    for root,cfg,regime in sources.values():
        saved=torch.load(root/'geometry.pt',map_location=device,weights_only=True)
        if not same_state(sim.state_dict(),saved):p.error('Saved geometry differs from the expected simulator geometry')
        for seed in seeds:
            for kind in kinds:
                if not (root/f'seed_{seed}'/f'{regime}_{kind}'/'best.pt').is_file():p.error(f'Missing checkpoint in {root}; review ZIPs exclude model weights')
    a.output.mkdir(parents=True,exist_ok=True)
    manifest={'evaluation':{k:str(v) if isinstance(v,Path) else [str(x) for x in v] if k=='results' else v for k,v in vars(a).items()},
              'selected_training_seeds':seeds,'device':str(device),'torch':torch.__version__,
              'paired_scenes':'Same evaluation generator seeds, batch size and scene count for every alpha/SNR/model; repeated conditions are not independent scenes.',
              'training_snr_reference':[0,10,20],'sources':[]}
    aggregate=[];per_seed=[];all_arrays={}
    with (a.output/'per_scene_errors.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['alpha','snr_db','method','training_seed','scene_id','nmse']);w.writeheader()
        for alpha,(root,cfg,regime) in sorted(sources.items()):
            bfile=root/f'bridge_{regime}.pt';b=torch.load(bfile,map_location=device,weights_only=True)
            bridge=PhysicsBridge(b['U'],b['T'],b['eigenvalues']).to(device).eval()
            models=[];files={str(bfile.resolve()):digest(bfile)}
            for seed in seeds:
                for kind in kinds:
                    file=root/f'seed_{seed}'/f'{regime}_{kind}'/'best.pt'
                    ck=torch.load(file,map_location=device,weights_only=True)
                    model=Reconstructor(bridge,kind=='hybrid',cfg['width']).to(device)
                    missing,unexpected=model.load_state_dict(ck['model'],strict=False)
                    if unexpected or any(not k.startswith('bridge.') for k in missing):raise ValueError(f'Checkpoint mismatch: {file}')
                    models.append((kind,seed,model.eval()));files[str(file.resolve())]=digest(file)
            manifest['sources'].append({'alpha':alpha,'regime':regime,'configuration':cfg,'sha256':files})
            for snr in a.snrs:
                values={('physics',-1):[]};values.update({(kind,seed):[] for kind,seed,_ in models})
                with torch.inference_mode():
                    for j,start in enumerate(range(0,a.scenes,a.batch_size)):
                        data=sim.batch(min(a.batch_size,a.scenes-start),a.evaluation_seed+j,alpha,snr)
                        predictions=[('physics',-1,bridge(data['x'],data['variance']))]
                        predictions += [(kind,seed,m(data['x'],data['variance'],data['scale'])) for kind,seed,m in models]
                        for kind,seed,pred in predictions:
                            errs=nmse(pred,data['y']).cpu().numpy()
                            if not np.isfinite(errs).all():raise ValueError('Nonfinite evaluation errors')
                            values[kind,seed].extend(errs.tolist())
                            w.writerows(dict(alpha=alpha,snr_db=snr,method=kind,training_seed=seed,scene_id=start+i,nmse=float(e)) for i,e in enumerate(errs))
                f.flush()
                for (kind,seed),v in values.items():
                    avg=float(np.mean(v));per_seed.append(dict(alpha=alpha,snr_db=snr,method=kind,training_seed=seed,
                        scenes=len(v),mean_nmse=avg,nmse_db=10*math.log10(max(avg,1e-30))))
                for kind in ['physics',*kinds]:
                    matrix=np.asarray([v for (k,s),v in values.items() if k==kind]);avg=float(matrix.mean())
                    all_arrays[alpha,snr,kind]=matrix
                    aggregate.append(dict(alpha=alpha,snr_db=snr,method=kind,scenes=a.scenes,model_count=len(matrix),
                        mean_nmse=avg,nmse_db=10*math.log10(max(avg,1e-30)),
                        snr_relation='trained' if snr in [0,10,20] else 'interpolation' if 0<snr<20 else 'extrapolation'))
                print(f'alpha={alpha:g}, SNR={snr:g}: complete ({a.scenes} paired scenes)',flush=True)
            del models,bridge
    gains=[]
    for alpha in sorted(sources):
        for snr in a.snrs:
            physics=float(all_arrays[alpha,snr,'physics'].mean());hybrid=float(all_arrays[alpha,snr,'hybrid'].mean())
            gains.append(dict(alpha=alpha,snr_db=snr,gain_db=10*math.log10(max(physics,1e-30)/max(hybrid,1e-30)),
                              error_reduction_percent=100*(1-hybrid/max(physics,1e-30))))
    write_csv(a.output/'summary.csv',aggregate);write_csv(a.output/'per_seed_summary.csv',per_seed);write_csv(a.output/'hybrid_gain.csv',gains)
    (a.output/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    render(aggregate,gains,a.output,a.fixed_snrs)
    (a.output/'README.txt').write_text('Outputs 01: NMSE vs SNR. Outputs 02: NMSE vs deformation. Output 03: hybrid gain.\n'
        'Linear plots use a linear y axis; dB plots use 10*log10(mean linear NMSE). Positive gain favors hybrid.\n'
        'Only trained amplitudes were evaluated. Lines connect measured points, not additional evaluated conditions.\n'
        'Physics has no training seed (-1 in CSV). Neural curves average linear errors over selected training seeds.\n'
        'All curves use matched propagation/noise draws. Scene pairing depends on evaluation seed and batch size.\n'
        'Training SNRs: 0,10,20 dB; other SNRs are interpolation or extrapolation, labelled in summary.csv.\n',encoding='utf-8')
    (a.output/'run_status.json').write_text(json.dumps({'completed':True}))
    print('\nSaved plots (PNG/PDF/SVG), summaries and per-scene errors to',a.output.resolve())
    print('To display in a notebook: from IPython.display import display, Image; display(Image(filename="'+str(a.output/'01_nmse_vs_snr_db.png')+'"))')


if __name__=='__main__':main()
