"""Inference on raw inverse-pilot channels, using a trusted trained checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from simulator import pack,unpack
from bridge import PhysicsBridge
from model import Reconstructor
from train import load_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('predictions.npz'))
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--regime',choices=['large','small'],default='large')
    p.add_argument('--kind',choices=['hybrid','direct'],default='hybrid')
    p.add_argument('--alpha',type=float,required=True)
    a=p.parse_args()
    config=json.loads((a.results/'configuration.json').read_text())
    if not np.isclose(a.alpha,config[f'{a.regime}_alpha'],rtol=0,atol=1e-9):
        raise ValueError('This checkpoint and bridge are tied to their training amplitude and geometry')
    if a.output.resolve()==a.input.resolve():raise ValueError('Use a separate output path')
    z=np.load(a.input,allow_pickle=False)
    observed=z['observations'];variance=z['noise_variance']
    if observed.ndim!=5 or observed.shape[1:]!=(8,25,25,2) or not np.iscomplexobj(observed):
        raise ValueError('observations must be complex [B,8,25,25,2]')
    if variance.shape!=(len(observed),8) or np.any(variance<=0) or not np.isfinite(variance).all() or not np.isfinite(observed).all():
        raise ValueError('noise_variance must be finite positive [B,8]; observations must be finite')
    if len(observed)==0:raise ValueError('Empty observation batch')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b=torch.load(a.results/f'bridge_{a.regime}.pt',map_location=device,weights_only=True)
    model=Reconstructor(PhysicsBridge(b['U'],b['T'],b['eigenvalues']),a.kind=='hybrid',config['width']).to(device)
    ck=torch.load(a.results/f'seed_{a.seed}'/f'{a.regime}_{a.kind}'/'best.pt',map_location=device,weights_only=True)
    load_state(model,ck['model']);model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(observed),32):
            raw=torch.tensor(observed[start:start+32],dtype=torch.complex64,device=device)
            var=torch.tensor(variance[start:start+32],dtype=torch.float32,device=device)
            scale=raw.abs().flatten(1).amax(1).clamp_min(1e-8)
            pred=model(pack(raw)/scale[:,None,None,None],var,scale)
            h=unpack(pred)[:,0]*scale[:,None,None,None]
            out.append(h.cpu().numpy())
    np.savez_compressed(a.output,H_hat=np.concatenate(out),alpha=a.alpha)
    print('Saved',a.output)


if __name__=='__main__':main()
