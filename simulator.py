"""Vectorized simulator; full-band noise power, two observed tones, no target scaling."""
from types import SimpleNamespace
import math
import numpy as np
import torch
from torch import nn
import reference_physics as ref


def pack(h):
    # h: B,M,25,25,2 -> B,4M,25,25, view-major Re0 Im0 Re1 Im1
    return torch.stack((h[...,0].real,h[...,0].imag,h[...,1].real,h[...,1].imag),dim=2).flatten(1,2)


def unpack(x):
    x=x.reshape(len(x),-1,4,25,25)
    return torch.stack((torch.complex(x[:,:,0],x[:,:,1]),torch.complex(x[:,:,2],x[:,:,3])),dim=-1)


class Simulator(nn.Module):
    def __init__(self,geometry_seed=2026,paths=3,pair=0):
        super().__init__()
        a=SimpleNamespace(fc=28e9,spacing=.125,NH_B=5,NV_B=5,NH_U=5,NV_U=5,M=8)
        pb,pu,zb,zu=ref.make_codebook(a,np.random.default_rng(geometry_seed))
        for name,value in zip(('pb','pu','zb','zu'),(pb,pu,zb,zu)):
            self.register_buffer(name,torch.tensor(value/(3e8/28e9),dtype=torch.float32))
        self.paths=paths;self.pair=pair

    def directions(self,b,l,generator):
        shape=(b,l)
        phi=(torch.rand(shape,device=self.pb.device,generator=generator)*2-1)*math.pi
        theta=(torch.rand(shape,device=self.pb.device,generator=generator)-.5)*math.pi
        return torch.stack((torch.sin(theta)*torch.cos(phi),torch.sin(theta)*torch.sin(phi),torch.cos(theta)),dim=-1)

    def atoms(self,db,du,alpha,rigid=False):
        # Input dirs B,L,3. Output B,M,25,25,L.
        pb=self.pb[None] if rigid else self.pb[None]+alpha*self.zb
        pu=self.pu[None] if rigid else self.pu[None]+alpha*self.zu
        ab=torch.exp(2j*math.pi*torch.einsum('mci,blc->bmil',pb,db))/5
        au=torch.exp(2j*math.pi*torch.einsum('mcj,blc->bmjl',pu,du))/5
        return ab[:,:,:,None,:]*au[:,:,None,:,:].conj()

    @torch.no_grad()
    def batch(self,n,seed,alpha,snr=None):
        g=torch.Generator(device=self.pb.device).manual_seed(seed)
        db=self.directions(n,self.paths,g);du=self.directions(n,self.paths,g)
        beta=torch.complex(torch.randn(n,self.paths,device=self.pb.device,generator=g),
                           torch.randn(n,self.paths,device=self.pb.device,generator=g))/math.sqrt(2*self.paths)
        delay=torch.rand(n,self.paths,device=self.pb.device,generator=g)
        phases=beta[:,None,:]*torch.exp(-2j*math.pi*torch.arange(32,device=self.pb.device)[None,:,None]/32*delay[:,None,:])
        # delay is fs*tau in [0,1]; same convention as the original generator.
        atoms=self.atoms(db,du,alpha)
        flat=atoms.flatten(2,3)
        spatial=torch.einsum('bmil,bmin->bmln',flat,flat.conj())/625
        temporal=torch.einsum('bkl,bkn->bln',phases,phases.conj())/32
        power=(spatial*temporal[:,None]).sum((-1,-2)).real.clamp_min(1e-12)
        tones=phases[:,self.pair:self.pair+2]
        clean=torch.einsum('bmijl,bkl->bmijk',atoms,tones)
        target=torch.einsum('bmijl,bkl->bmijk',self.atoms(db,du,alpha,True),tones)
        if snr is None:
            # Balance across batches; random permutation does not change noise pairing between arms.
            snrs=torch.tensor([0.,10.,20.],device=self.pb.device)[(torch.arange(n,device=self.pb.device)+seed%3)%3]
        else:
            snrs=torch.as_tensor(snr,dtype=power.dtype,device=self.pb.device)
            if snrs.ndim==0:
                snrs=snrs.expand(n)
            if snrs.shape!=(n,) or not torch.isfinite(snrs).all():
                raise ValueError('SNR must be a finite scalar or a vector with one value per scene')
        variance=power/10**(snrs[:,None]/10)
        # Inverse unitary pilots preserve iid complex Gaussian noise. Simulate directly after inversion.
        noise=torch.complex(torch.randn(clean.shape,device=self.pb.device,generator=g),
                            torch.randn(clean.shape,device=self.pb.device,generator=g))*torch.sqrt(variance[:,:,None,None,None]/2)
        observed=clean+noise
        scale=observed.abs().flatten(1).amax(1).clamp_min(1e-8)
        x=pack(observed)/scale[:,None,None,None]
        y=pack(target)/scale[:,None,None,None]
        energy=target.abs().square().sum((2,3,4))
        rho=((clean.conj()*target).sum((2,3,4)).abs()/torch.sqrt(clean.abs().square().sum((2,3,4))*energy).clamp_min(1e-12)).mean(1)
        difference=((clean-target).abs().square().sum((2,3,4))/energy.clamp_min(1e-12)).mean(1)
        return {'x':x,'y':y,'variance':variance,'scale':scale,'snr':snrs,'rho':rho,'difference':difference}
