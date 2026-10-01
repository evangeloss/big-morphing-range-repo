"""ONE network trained jointly on all specified known b/lambda values.

Each mixed minibatch contains equal counts from every amplitude. Fixed physics
operators depend on geometry; neural weights, optimizer and checkpoint are shared.
"""
import argparse,csv,json,math,os,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from simulator import Simulator
from bridge import PhysicsBridge,build_bridge
from shared_b_model import SharedReconstructor
from model import nmse
from evaluate_sweep import render


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def save(obj,path):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)


def cpu_state(model):return {k:v.detach().cpu() for k,v in model.state_dict().items()}


@torch.no_grad()
def dataset_batch(sim,bridge,n,seed,alpha,snr=None):
    d=sim.batch(n,seed,alpha,snr)
    d['base']=bridge(d['x'],d['variance'])
    d['alpha']=torch.full((n,),alpha,device=d['x'].device)
    return d


def predict(model,d):return model(d['x'],d['base'],d['variance'],d['scale'],d['alpha'])


def training_snr_batch(values,n,step,alpha_index,device):
    # Rotate through every SNR for every amplitude; counts differ by at most one.
    levels=torch.tensor(values,dtype=torch.float32,device=device)
    return levels[(torch.arange(n,device=device)+step*n+alpha_index)%len(values)]


@torch.no_grad()
def assess(model,sim,bridge,alpha,snr,n,batch,seed):
    model.eval();neural=[];physics=[]
    for j,start in enumerate(range(0,n,batch)):
        d=dataset_batch(sim,bridge,min(batch,n-start),seed+j,alpha,snr)
        neural.extend(nmse(predict(model,d),d['y']).cpu().tolist())
        physics.extend(nmse(d['base'],d['y']).cpu().tolist())
    return np.asarray(neural),np.asarray(physics)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('shared_b_results'))
    p.add_argument('--alphas',nargs='+',type=float,default=[.01,.05,.1,.25,.4,.6,.75,.9])
    p.add_argument('--seeds',nargs='+',type=int,default=[11])
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--validation-scenes',type=int,default=64,help='Scenes per amplitude/SNR validation condition')
    p.add_argument('--test-scenes',type=int,default=500)
    p.add_argument('--training-snrs',nargs='+',type=float,default=[-10,-5,0,5,10,15,20],help='SNRs used for both training and validation')
    p.add_argument('--snrs',nargs='+',type=float,default=[-10,-5,0,5,10,15,20],help='Independent test SNRs')
    p.add_argument('--atoms',type=int,default=1024)
    p.add_argument('--rank',type=int,default=384)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--quick',action='store_true')
    a=p.parse_args()
    if a.quick:
        a.epochs=2;a.steps=4;a.batch_size=2*len(a.alphas);a.validation_scenes=4;a.test_scenes=8;a.atoms=32;a.rank=16;a.width=8;a.seeds=[11]
    if (len(set(a.alphas))!=len(a.alphas) or any(not math.isfinite(v) or v<=0 for v in a.alphas)
        or a.batch_size%len(a.alphas) or min(a.epochs,a.steps,a.batch_size,a.validation_scenes,a.test_scenes,a.atoms,a.rank,a.width)<1
        or a.width%8 or len(set(a.seeds))!=len(a.seeds) or not all(0<=s<999 for s in a.seeds)
        or a.steps*a.epochs*len(a.alphas)>=1000000 or not {0,10,20}.issubset(a.snrs)
        or len(set(a.training_snrs))!=len(a.training_snrs)
        or not all(math.isfinite(v) for v in a.snrs+a.training_snrs)):
        p.error('Check settings: batch size must be divisible by the number of amplitudes; include SNRs 0,10,20')
    a.snrs=sorted(set(a.snrs));a.training_snrs=sorted(a.training_snrs);out=a.output.resolve()
    config={k:v for k,v in vars(a).items() if k not in ['output','resume','evaluate_only','snrs','test_scenes']}
    config.update({'architecture':'one_shared_hybrid_with_known_alpha','paths':3,'geometry_seed':2026,'pair_start':0,
                   'snr_sampling':'balanced_rotation_v1','selection':'macro mean linear validation NMSE over all amplitude/SNR conditions'})
    if out.exists() and any(out.iterdir()):
        if not (a.resume or a.evaluate_only):p.error('Use a new output directory or --resume')
        if json.loads((out/'shared_configuration.json').read_text())!=config:p.error('Training settings differ from saved shared model')
    elif a.evaluate_only:p.error('Evaluation-only needs an existing trained shared model')
    out.mkdir(parents=True,exist_ok=True)
    (out/'shared_configuration.json').write_text(json.dumps(config,indent=2))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    sim=Simulator().to(device);torch.save(sim.state_dict(),out/'geometry.pt')
    banks={};info=[]
    for i,alpha in enumerate(a.alphas):
        file=out/f'physics_{i}.pt'
        if file.exists():
            b=torch.load(file,map_location=device,weights_only=True)
            bridge=PhysicsBridge(b['U'],b['T'],b['eigenvalues']);details=b['info']
        else:
            print(f'Building fixed physics operator for b/lambda={alpha:g} (not a trained neural network)',flush=True)
            bridge,details=build_bridge(sim,alpha,a.atoms,a.rank)
            torch.save({**{k:v.cpu() for k,v in bridge.state_dict().items()},'info':details},file)
        banks[alpha]=bridge.to(device);info.append(details)
    (out/'physics_diagnostics.json').write_text(json.dumps(info,indent=2))
    summaries=[];scene_rows=[];global_history=[]
    for seed in a.seeds:
        folder=out/f'seed_{seed}';folder.mkdir(exist_ok=True)
        torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
        model=SharedReconstructor(a.width).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=8,min_lr=3e-6)
        start=0;best=float('inf');history=[];by_condition=[]
        def validate(epoch):
            vals=[]
            for alpha in a.alphas:
                for snr in a.training_snrs:
                    v,_=assess(model,sim,banks[alpha],alpha,snr,a.validation_scenes,a.batch_size,3000000000+seed*10000)
                    value=float(v.mean());vals.append(value)
                    by_condition.append(dict(epoch=epoch,alpha=alpha,snr_db=snr,validation_nmse=value))
            return float(np.mean(vals))
        if a.resume and (folder/'last.pt').exists():
            c=torch.load(folder/'last.pt',map_location=device,weights_only=False)
            model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);scheduler.load_state_dict(c['scheduler'])
            start=c['epoch'];best=c['best'];history=c['history'];by_condition=c['by_condition']
        elif not a.evaluate_only:
            best=validate(0);save({'model':cpu_state(model),'epoch':0,'validation_nmse':best},folder/'best.pt')
        if not a.evaluate_only:
            per_alpha=a.batch_size//len(a.alphas)
            for epoch in range(start,a.epochs):
                clock=time.time();model.train();total=0.
                for step in range(a.steps):
                    # Equal examples from EVERY amplitude in EVERY optimization step.
                    groups=[dataset_batch(sim,banks[alpha],per_alpha,
                        1000000+seed*1000000+(epoch*a.steps+step)*len(a.alphas)+i,alpha,
                        training_snr_batch(a.training_snrs,per_alpha,epoch*a.steps+step,i,device))
                        for i,alpha in enumerate(a.alphas)]
                    d={k:torch.cat([g[k] for g in groups]) for k in ['x','y','base','variance','scale','alpha']}
                    opt.zero_grad(set_to_none=True);loss=nmse(predict(model,d),d['y']).mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())
                val=validate(epoch+1)
                if not math.isfinite(val):raise RuntimeError('Nonfinite validation loss')
                scheduler.step(val)
                if val<best:
                    best=val;save({'model':cpu_state(model),'epoch':epoch+1,'validation_nmse':best},folder/'best.pt')
                history.append(dict(epoch=epoch+1,train_nmse=total/a.steps,validation_nmse=val,learning_rate=opt.param_groups[0]['lr'],seconds=time.time()-clock))
                write_csv(folder/'history.csv',history);write_csv(folder/'validation_by_condition.csv',by_condition)
                save({'model':cpu_state(model),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                      'epoch':epoch+1,'best':best,'history':history,'by_condition':by_condition},folder/'last.pt')
                print(f'Seed {seed}: shared model epoch {epoch+1}/{a.epochs}, train={total/a.steps:.5g}, macro validation={val:.5g}',flush=True)
        ck=torch.load(folder/'best.pt',map_location=device,weights_only=True);model.load_state_dict(ck['model'])
        if not history and (folder/'history.csv').exists():history=list(csv.DictReader((folder/'history.csv').open()))
        global_history.extend(dict(seed=seed,**r) for r in history)
        for alpha in a.alphas:
            for snr in a.snrs:
                # Identical test latent scenes/noise draws across alpha/SNR/training-seed comparisons.
                neural,physics=assess(model,sim,banks[alpha],alpha,snr,a.test_scenes,a.batch_size,4000000000)
                for kind,v in [('hybrid',neural),('physics',physics)]:
                    if kind=='physics' and seed!=a.seeds[0]:continue
                    mean=float(v.mean())
                    summaries.append(dict(alpha=alpha,snr_db=snr,method=kind,training_seed=seed if kind=='hybrid' else -1,
                        mean_nmse=mean,nmse_db=10*np.log10(max(mean,1e-30)),best_epoch=ck['epoch'] if kind=='hybrid' else 0))
                    scene_rows.extend(dict(alpha=alpha,snr_db=snr,method=kind,training_seed=seed if kind=='hybrid' else -1,scene_id=i,nmse=float(e)) for i,e in enumerate(v))
                print(f'One shared checkpoint: alpha={alpha:g}, SNR={snr:g} evaluated',flush=True)
        del model,opt,scheduler
    evaluation=out/'evaluation';evaluation.mkdir(exist_ok=True)
    write_csv(evaluation/'per_seed_summary.csv',summaries);write_csv(evaluation/'per_scene_errors.csv',scene_rows)
    aggregate=[];gains=[]
    for alpha in a.alphas:
        for snr in a.snrs:
            means={}
            for kind in ['physics','hybrid']:
                values=[r['mean_nmse'] for r in summaries if r['alpha']==alpha and r['snr_db']==snr and r['method']==kind]
                mean=float(np.mean(values));means[kind]=mean
                aggregate.append(dict(alpha=alpha,snr_db=snr,method=kind,mean_nmse=mean,nmse_db=10*np.log10(max(mean,1e-30))))
            gains.append(dict(alpha=alpha,snr_db=snr,gain_db=10*np.log10(means['physics']/means['hybrid']),
                              error_reduction_percent=100*(1-means['hybrid']/means['physics'])))
    write_csv(evaluation/'summary.csv',aggregate);write_csv(evaluation/'hybrid_gain.csv',gains)
    render(aggregate,gains,evaluation,[s for s in a.training_snrs if s in a.snrs])
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,6))
    for seed in a.seeds:
        h=[r for r in global_history if int(r['seed'])==seed]
        for field,label,style in [('train_nmse','Train','-'),('validation_nmse','Macro validation','--')]:
            ax.plot([int(r['epoch']) for r in h],10*np.log10([float(r[field]) for r in h]),style,label=f'{label}, seed {seed}')
    ax.set(xlabel='Epoch',ylabel='Mean NMSE (dB)',title=f'One shared network — {len(a.alphas)} amplitudes, {len(a.training_snrs)} training SNRs')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(evaluation/f'04_shared_learning_history.{ext}',dpi=180)
    plt.close(fig)
    report=['# One shared hybrid model',f'Trained amplitudes b/lambda: {a.alphas}',f'Training seeds: {a.seeds}',
        'One neural checkpoint per seed is used at EVERY amplitude. Physics matrices are fixed geometry-dependent operators, not separate neural networks.',
        'Balanced amplitudes in every batch; validation selects one checkpoint by macro mean linear NMSE over all amplitudes and training SNRs.',
        f'Training and validation SNRs: {a.training_snrs} dB. Test SNRs: {a.snrs} dB. All test deformation amplitudes were trained.',
        'See evaluation/summary.csv and figures. Known geometry/noise/prior assumptions remain unchanged.']
    (out/'SHARED_MODEL.md').write_text('\n'.join(report),encoding='utf-8')
    (out/'sweep_status.json').write_text(json.dumps({'completed':True,'model_type':'one_shared_neural_network_per_seed','evaluation_folder':str(evaluation)}))
    print('\n'.join(report))


if __name__=='__main__':main()
