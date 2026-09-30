"""Scientific and implementation checks. Run before expensive training."""
import math
import numpy as np
import torch
from simulator import Simulator,pack,unpack
from bridge import build_bridge
from model import Reconstructor,nmse
import reference_physics as ref


def main():
    torch.set_num_threads(4)
    sim=Simulator()
    angles=[.4,.7,-.2,.5]
    phi,theta,phi_u,theta_u=angles
    direction=lambda p,t: [math.sin(t)*math.cos(p),math.sin(t)*math.sin(p),math.cos(t)]
    db=torch.tensor([[direction(phi,theta)]],dtype=torch.float32)
    du=torch.tensor([[direction(phi_u,theta_u)]],dtype=torch.float32)
    actual=sim.atoms(db,du,.5)[0,0,:,:,0].numpy()
    params=dict(BETA=np.array([1+0j]),delay=np.array([0.]),AOA_az=np.array([phi]),AOA_el=np.array([theta]),
                DOA_az=np.array([phi_u]),DOA_el=np.array([theta_u]))
    expected=ref.build_H_fim_from_paths(params,sim.pb.numpy(),.5*sim.zb[0].numpy(),sim.pu.numpy(),.5*sim.zu[0].numpy(),1.,1.,32)[:,:,0]
    np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-5)
    a=sim.batch(16,771,0,20);b=sim.batch(16,771,.5,20)
    # Independent full-band construction validates the efficient noise-power identity.
    g=torch.Generator().manual_seed(771)
    db_check=sim.directions(16,3,g);du_check=sim.directions(16,3,g)
    beta=torch.complex(torch.randn(16,3,generator=g),torch.randn(16,3,generator=g))/math.sqrt(6)
    delay=torch.rand(16,3,generator=g)
    phase=beta[:,None,:]*torch.exp(-2j*math.pi*torch.arange(32)[None,:,None]/32*delay[:,None,:])
    full=torch.einsum('bmijl,bkl->bmijk',sim.atoms(db_check,du_check,.5),phase)
    torch.testing.assert_close(full.abs().square().mean((2,3,4))/100,b['variance'],rtol=1e-5,atol=1e-10)
    z=unpack(a['x']);torch.testing.assert_close(pack(z),a['x'])
    ya=a['y']*a['scale'][:,None,None,None];yb=b['y']*b['scale'][:,None,None,None]
    torch.testing.assert_close(ya,yb,rtol=1e-5,atol=1e-7)
    assert a['rho'].min()>.99999 and a['difference'].max()<1e-10
    observed=unpack(a['x'])*a['scale'][:,None,None,None,None]
    target=unpack(a['y'])*a['scale'][:,None,None,None,None]
    empirical=(observed-target).abs().square().mean((2,3,4))
    ratio=float((empirical/a['variance']).mean());assert .95<ratio<1.05,ratio
    # With rigid arrays at alpha=0, tone-averaged per-view power is identical across views.
    torch.testing.assert_close(a['variance'],a['variance'][:,:1].expand_as(a['variance']))
    bridge,info=build_bridge(sim,.5,atoms=32,max_rank=32)
    # Compare the transport with an independent dual ridge solve for the same dictionary.
    g=torch.Generator().manual_seed(314159)
    db=sim.directions(32,1,g);du=sim.directions(32,1,g)
    D=sim.atoms(db,du,.5).squeeze(-1).reshape(32,5000).T/math.sqrt(32)
    D0=sim.atoms(db,du,.5,True).squeeze(-1).reshape(32,625).T/math.sqrt(32)
    obs=unpack(b['x'][:1]).permute(0,4,1,2,3).reshape(2,5000).T
    sigma=b['variance'][0].mean()
    result=D0@torch.linalg.solve(D.conj().T@D+sigma*torch.eye(32),D.conj().T@obs)
    direct=pack(result.T.reshape(1,2,1,25,25).permute(0,2,3,4,1))
    torch.testing.assert_close(bridge(b['x'][:1],b['variance'][:1]),direct,rtol=2e-3,atol=2e-4)
    model=Reconstructor(bridge,True,8)
    pred=model(b['x'][:2],b['variance'][:2],b['scale'][:2])
    torch.testing.assert_close(pred,bridge(b['x'][:2],b['variance'][:2]))
    loss=nmse(pred,b['y'][:2]).mean();loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print('PASS: reference steering, packing, paired targets, full-band noise power, noise calibration, bridge algebra, initial skip and gradients.')
    print('Bridge test:',info)


if __name__=='__main__':main()
