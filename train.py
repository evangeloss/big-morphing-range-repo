"""Physics transport + learned refinement, trained at a fixed large deformation."""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from simulator import Simulator,unpack,pack
from bridge import build_bridge,PhysicsBridge
from model import Reconstructor,nmse


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def state(model):return {k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('bridge.')}


def load_state(model,s):
    missing,unexpected=model.load_state_dict(s,strict=False)
    assert not unexpected and all(k.startswith('bridge.') for k in missing)


def save_checkpoint(obj,path):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


@torch.no_grad()
def evaluate(model,sim,alpha,n,batch,seed,snr=None):
    model.eval();values=[];metadata=[]
    for j,start in enumerate(range(0,n,batch)):
        data=sim.batch(min(batch,n-start),seed+j,alpha,snr)
        prediction=model(data['x'],data['variance'],data['scale'])
        err=nmse(prediction,data['y'])
        values.extend(err.cpu().tolist())
        metadata.extend({'sample':start+i,'snr_db':float(data['snr'][i]),'rho':float(data['rho'][i]),
                         'clean_difference':float(data['difference'][i])} for i in range(len(err)))
    return np.asarray(values),metadata


class Baseline(torch.nn.Module):
    def __init__(self,bridge,kind):super().__init__();self.bridge=bridge;self.kind=kind
    def forward(self,x,v,s):
        if self.kind=='physics':return self.bridge(x,v)
        if self.kind=='first_view':return x[:,:4]
        return pack(unpack(x).mean(1,keepdim=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--train-only',action='store_true',help='Train and validate; save models without final test evaluation')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--large-alpha',type=float,default=.5)
    p.add_argument('--small-alpha',type=float,default=.02)
    p.add_argument('--seeds',nargs='+',type=int,default=[11])
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--validation-scenes',type=int,default=256)
    p.add_argument('--test-scenes',type=int,default=300)
    p.add_argument('--atoms',type=int,default=1024)
    p.add_argument('--rank',type=int,default=384)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--paths',type=int,default=3)
    p.add_argument('--geometry-seed',type=int,default=2026)
    p.add_argument('--pair-start',type=int,default=0)
    p.add_argument('--gap-target-db',type=float,default=2.)
    a=p.parse_args()
    if a.quick:
        a.epochs=2;a.steps=2;a.batch_size=4;a.validation_scenes=8;a.test_scenes=8;a.atoms=32;a.rank=16;a.width=8;a.seeds=[11]
    if (min(a.epochs,a.steps,a.batch_size,a.validation_scenes,a.test_scenes,a.atoms,a.rank,a.paths)<=0
        or a.width%8 or a.width<8 or not 0<=a.pair_start<31 or not 0<=a.small_alpha<a.large_alpha
        or not all(math.isfinite(v) for v in (a.small_alpha,a.large_alpha,a.gap_target_db))
        or not all(0<=s<1000 for s in a.seeds) or len(set(a.seeds))!=len(a.seeds)
        or a.epochs*a.steps>=100000):p.error('Invalid dimensions, seeds, amplitudes, or >=100000 training batches/seed')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('resume','output')}
    if a.output.exists() and any(a.output.iterdir()):
        if not a.resume:p.error('Output exists; use --resume or a new output directory')
        if json.loads((a.output/'configuration.json').read_text())!=config:p.error('Resume requires identical experiment settings')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'configuration.json').write_text(json.dumps(config,indent=2))
    (a.output/'run_status.json').write_text(json.dumps({'completed':False}))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    (a.output/'environment.json').write_text(json.dumps({'device':str(device),'torch':torch.__version__,'numpy':np.__version__},indent=2))
    sim=Simulator(a.geometry_seed,a.paths,a.pair_start).to(device)
    torch.save(sim.state_dict(),a.output/'geometry.pt')
    bridges={};infos=[]
    for label,alpha in [('large',a.large_alpha),('small',a.small_alpha)]:
        file=a.output/f'bridge_{label}.pt'
        if file.exists():
            saved=torch.load(file,map_location=device,weights_only=True)
            bridge=PhysicsBridge(saved['U'],saved['T'],saved['eigenvalues']);info=saved['info']
        else:
            print(f'Building physics bridge for alpha={alpha}',flush=True)
            bridge,info=build_bridge(sim,alpha,a.atoms,a.rank)
            torch.save({**{k:v.cpu() for k,v in bridge.state_dict().items()},'info':info},file)
        bridges[label]=bridge.to(device);infos.append(info)
        print(info,flush=True)
    (a.output/'bridge_diagnostics.json').write_text(json.dumps(infos,indent=2))
    summaries=[];errors=[];parameters={}
    for seed in a.seeds:
        for label,alpha in [('large',a.large_alpha),('small',a.small_alpha)]:
            for kind in ['hybrid','direct']:
                arm=f'{label}_{kind}';folder=a.output/f'seed_{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
                torch.manual_seed(seed)
                if device.type=='cuda':torch.cuda.manual_seed_all(seed)
                model=Reconstructor(bridges[label],kind=='hybrid',a.width).to(device)
                parameters[arm]=sum(t.numel() for t in model.parameters())
                opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=8,min_lr=3e-6)
                start=0;best=float('inf');best_epoch=0;stale=0;history=[];finished=False
                last=folder/'last.pt'
                if a.resume and last.exists():
                    ck=torch.load(last,map_location=device,weights_only=False)
                    load_state(model,ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
                    start=ck['epoch'];best=ck['best'];best_epoch=ck['best_epoch'];stale=ck['stale'];history=ck['history'];finished=ck['finished']
                else:
                    initial,_=evaluate(model,sim,alpha,a.validation_scenes,a.batch_size,1000000000+seed*10000)
                    best=float(initial.mean())
                    save_checkpoint({'model':state(model),'epoch':0,'validation_nmse':best},folder/'best.pt')
                    (folder/'initial_validation.json').write_text(json.dumps({'nmse':best}))
                for epoch in range(start,a.epochs) if not finished else []:
                    model.train();total=0.;clock=time.time()
                    for step in range(a.steps):
                        # All four arms receive matching latent scenes and standardized noise.
                        data=sim.batch(a.batch_size,1000000+seed*100000+epoch*a.steps+step,alpha)
                        opt.zero_grad(set_to_none=True)
                        prediction=model(data['x'],data['variance'],data['scale'])
                        loss=nmse(prediction,data['y']).mean()
                        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss: {arm}, epoch {epoch}')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        opt.step();total+=float(loss.detach())
                    vals,_=evaluate(model,sim,alpha,a.validation_scenes,a.batch_size,1000000000+seed*10000)
                    val=float(vals.mean())
                    if not math.isfinite(val):raise RuntimeError('Nonfinite validation error')
                    scheduler.step(val)
                    if val<best:
                        best=val;best_epoch=epoch+1;stale=0
                        save_checkpoint({'model':state(model),'epoch':best_epoch,'validation_nmse':best},folder/'best.pt')
                    else:stale+=1
                    # At least 50 epochs, then stop only after sustained validation stagnation and LR reductions.
                    finished=(epoch+1==a.epochs) or (epoch+1>=50 and stale>=25 and opt.param_groups[0]['lr']<=3.75e-5)
                    history.append(dict(epoch=epoch+1,train_nmse=total/a.steps,validation_nmse=val,
                                        learning_rate=opt.param_groups[0]['lr'],seconds=time.time()-clock))
                    write_csv(folder/'history.csv',history)
                    save_checkpoint({'model':state(model),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                        'epoch':epoch+1,'best':best,'best_epoch':best_epoch,'stale':stale,'history':history,'finished':finished},last)
                    print(f'{seed} {arm} {epoch+1}/{a.epochs}: train={total/a.steps:.5g}, val={val:.5g}, lr={opt.param_groups[0]["lr"]:.3g}',flush=True)
                    if finished:break
                if a.train_only:
                    del model,opt,scheduler
                    continue
                best_saved=torch.load(folder/'best.pt',map_location=device,weights_only=True)
                load_state(model,best_saved['model'])
                for snr in [0,10,20]:
                    values,meta=evaluate(model,sim,alpha,a.test_scenes,a.batch_size,2000000000+seed*10000,snr)
                    summaries.append({'seed':seed,'arm':arm,'alpha':alpha,'snr_db':snr,'mean_nmse':float(values.mean()),
                        'nmse_db':float(10*np.log10(max(values.mean(),1e-30))),'p90_nmse':float(np.quantile(values,.9)),
                        'best_epoch':best_saved['epoch'],'trained_epochs':len(history)})
                    errors.extend(dict(seed=seed,arm=arm,alpha=alpha,model_nmse=float(v),**m) for v,m in zip(values,meta))
                write_csv(a.output/'summary.csv',summaries);write_csv(a.output/'per_scene_errors.csv',errors)
                del model,opt,scheduler
            if a.train_only:continue
            for kind in ['physics','mean_view','first_view']:
                model=Baseline(bridges[label],kind)
                for snr in [0,10,20]:
                    values,meta=evaluate(model,sim,alpha,a.test_scenes,a.batch_size,2000000000+seed*10000,snr)
                    arm=f'{label}_{kind}'
                    summaries.append({'seed':seed,'arm':arm,'alpha':alpha,'snr_db':snr,'mean_nmse':float(values.mean()),
                        'nmse_db':float(10*np.log10(max(values.mean(),1e-30))),'p90_nmse':float(np.quantile(values,.9)),
                        'best_epoch':0,'trained_epochs':0})
                    errors.extend(dict(seed=seed,arm=arm,alpha=alpha,model_nmse=float(v),**m) for v,m in zip(values,meta))
    if a.train_only:
        (a.output/'parameter_counts.json').write_text(json.dumps(parameters,indent=2))
        report=['# Training completed — final test evaluation not run',
                f'Seeds: {a.seeds}. Large deformation: {a.large_alpha}. Small reference: {a.small_alpha}.',
                'Four models per seed were trained from fresh initialization unless --resume was explicitly supplied.',
                'Validation selected best.pt and controlled the learning rate; no test scenes were evaluated.',
                'Keep the entire results directory, including bridge_*.pt, geometry.pt and seed_* checkpoints.',
                'Use evaluate_sweep.py later to generate NMSE comparisons.']
        (a.output/'TRAINING_COMPLETE.md').write_text('\n'.join(report),encoding='utf-8')
        (a.output/'run_status.json').write_text(json.dumps({'completed':True,'training_completed':True,'test_evaluation_completed':False}))
        print('\n'.join(report))
        return
    write_csv(a.output/'summary.csv',summaries);write_csv(a.output/'per_scene_errors.csv',errors)
    (a.output/'parameter_counts.json').write_text(json.dumps(parameters,indent=2))
    rng=np.random.default_rng(712);gaps=[]
    for seed in a.seeds:
        for kind in ['hybrid','direct','physics']:
            for snr in [0,10,20]:
                def get(label):return np.array([r['model_nmse'] for r in errors if r['seed']==seed and r['arm']==f'{label}_{kind}' and r['snr_db']==snr])
                large,small=get('large'),get('small');ids=rng.integers(len(large),size=(1000,len(large)))
                boot=10*np.log10(large[ids].mean(1)/small[ids].mean(1))
                gap=float(10*np.log10(large.mean()/small.mean()))
                gaps.append(dict(seed=seed,kind=kind,snr_db=snr,gap_db=gap,ci_low=float(np.quantile(boot,.025)),
                                 ci_high=float(np.quantile(boot,.975)),within_requested_gap=gap<=a.gap_target_db))
    write_csv(a.output/'deformation_gap.csv',gaps)
    lines=['# Large deformation reconstruction',f'Quick smoke run: {a.quick}. Seeds: {a.seeds}.',
           f'Large alpha={a.large_alpha}; matched small alpha={a.small_alpha}. Desired gap <= {a.gap_target_db} dB.',
           'Scores below pool equal-size test sets across seeds before converting mean NMSE to dB.',
           '| Model | SNR | Large NMSE dB | Small NMSE dB | Gap dB |','|---|---|---|---|---|']
    for kind in ['hybrid','direct','physics','mean_view','first_view']:
        for snr in [0,10,20]:
            def score(label):return 10*np.log10(np.mean([r['mean_nmse'] for r in summaries if r['arm']==f'{label}_{kind}' and r['snr_db']==snr]))
            large,small=score('large'),score('small')
            lines.append(f'| {kind} | {snr} | {large:.3f} | {small:.3f} | {large-small:.3f} |')
    lines+=['','Positive gap means large deformation is worse. Inspect both absolute errors and the gap.',
            'One-seed results are preliminary. Paired scene intervals are conditional on each trained model and do not measure training-seed uncertainty.',
            'Prior, geometry and receiver noise variances are assumed known. No test-scene angles/gains/delays enter the estimator.',
            'The prior transport uses a finite-rank approximation and mean noise variance across views; it is not an oracle.',
            'Fresh training scenes each step; trained epochs can differ due to validation stopping. Check learning curves and bridge diagnostics.']
    (a.output/'PASTE_BACK.md').write_text('\n'.join(lines),encoding='utf-8')
    (a.output/'run_status.json').write_text(json.dumps({'completed':True}))
    print('\n'.join(lines))


if __name__=='__main__':main()
