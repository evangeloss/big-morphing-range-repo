"""Reduced-rank LMMSE transport based only on geometry and an angle prior."""
import math
import torch
from torch import nn
from simulator import pack,unpack


class PhysicsBridge(nn.Module):
    def __init__(self,U,T,eigenvalues):
        super().__init__()
        self.register_buffer('U',U)
        self.register_buffer('T',T)
        self.register_buffer('eigenvalues',eigenvalues)

    def forward(self,x,variance):
        # The common-variance approximation averages the eight calibrated view variances.
        obs=unpack(x).permute(0,4,1,2,3).reshape(len(x),2,5000)
        coefficients=obs@self.U.conj()
        shrink=self.eigenvalues[None,:]/(self.eigenvalues[None,:]+variance.mean(1)[:,None])
        estimate=(coefficients*shrink[:,None,:])@self.T.T
        return pack(estimate.reshape(len(x),2,1,25,25).permute(0,2,3,4,1))


@torch.no_grad()
def build_bridge(sim,alpha,atoms=1024,max_rank=384,seed=314159):
    # These directions are numerical quadrature samples, never scene truth.
    g=torch.Generator(device=sim.pb.device).manual_seed(seed)
    db=sim.directions(atoms,1,g);du=sim.directions(atoms,1,g)
    D=sim.atoms(db,du,alpha).squeeze(-1).reshape(atoms,5000).T/math.sqrt(atoms)
    D0=sim.atoms(db,du,alpha,True).squeeze(-1).reshape(atoms,625).T/math.sqrt(atoms)
    gram=D.conj().T@D
    values,V=torch.linalg.eigh((gram+gram.conj().T)/2)
    order=torch.argsort(values,descending=True)
    values=values[order];V=V[:,order]
    positive=values>values[0]*1e-6
    rank=min(max_rank,int(positive.sum()))
    retained=float(values[:rank].sum()/values.clamp_min(0).sum())
    values=values[:rank].clamp_min(1e-12);V=V[:,:rank]
    U=(D@V)/values.sqrt()[None]
    T=(D0@V)/values.sqrt()[None]
    info={'alpha':alpha,'quadrature_atoms':atoms,'rank':rank,'prior_energy_retained':retained,
          'mean_view_noise_approximation':True,'dictionary_seed':seed}
    return PhysicsBridge(U,T,values),info
