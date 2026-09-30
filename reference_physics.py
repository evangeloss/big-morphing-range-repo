#!/usr/bin/env python3
"""Estimability experiment adapted from test-for-deformation-channel-visual (2).

Requires Python >=3.10 and NumPy. Optional plots require matplotlib.
    python channel_estimability.py --quick
    python channel_estimability.py --samples 200 --diagnostic-samples 5 --plot

Conventions retained from notebook cells 1,3,5,6,8,15,16,39,46:
H=(N_B,N_U,K); positions=(3,N); Y=H@S; fc=28 GHz, fs=100 kHz;
5x5 arrays, lambda/8 spacing, K=32, M=8, Tpilots=1; delay~U[0,1/fs].
AOA labels apply to BS, DOA to UE. Despite the name *_el, the notebook
uses sin(theta)cos(phi),sin(theta)sin(phi),cos(theta): POLAR-angle formula
with theta drawn from [-pi/2,pi/2]. This is preserved, not corrected silently.
This is a far-field, phase-only model, without coupling, visibility changes,
path loss, or deformation-dependent path gains. It does not validate hardware.

Diagnostics condition on known, fixed geometry and pilots. Paths are shared
across views and across alpha for paired comparisons; independent across samples.
No target-based normalization or per-channel unit-norm preprocessing is used.
Full unitary pilots are invertible; they are not compressed measurements.
Each additional view/pilot adds energy and observations (no fixed-budget claim).

The gain oracle KNOWS angles/delays, but estimates gains from observations.
The local Jacobian includes all 7L real latent parameters. Delays are scaled
as fs*tau and gains by sqrt(L), avoiding seconds/radians conditioning artifacts.
Target estimability is tested via J_H null(J_Y), not parameter rank alone.
A target CRB is reported only when that null-space test passes. It is a local,
fixed-noise-covariance bound, not a proof of global identifiability or CNN accuracy.
Noise covariance is frozen at the nominal channel when differentiating, even
though the simulation follows the notebook's per-view signal-power SNR rule.
Reference: https://www.nokia.com/bell-labs/publications-and-media/publications/
parameter-estimation-problems-with-singular-information-matrices/

By default diagnostics use full OFDM data. --pair-start k selects ONLY tones
k,k+1 for observations AND reference targets, retaining the original K in the
OFDM phase denominator. Noise is still calibrated/generated on the full band
before slicing, exactly as in the notebook. Do not set K=2 to select a pair.
The pair is fixed across independent channel samples; overlapping pairs are
not counted as independent realizations. Run separate pair starts to check
frequency dependence. Geometry/pilots remain known; this is not CNN training
and does not reproduce the notebook's target-dependent input normalization.
Geometry distance is zero within each fixed-codebook experiment. A sparse random
dataset cannot prove absence of collisions. Inspect the reported neighbor input
distance before interpreting its target distance. Low correlation alone does
not establish non-estimability. No automatic universal 'good dataset' threshold.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from metrics import describe


def require(ok, message):
    if not ok:
        raise ValueError(message)


def energy(x):
    return float(np.vdot(x, x).real)


def nmse(x, reference):
    den = energy(reference)
    require(den > np.finfo(float).tiny, 'Zero/underflow reference energy')
    return energy(x - reference) / den


def realvec(x):
    x = np.asarray(x).ravel()
    return np.concatenate((x.real, x.imag))


def generate_path_parameters(L, fc, fs, rng):
    p = {key: rng.uniform(-np.pi, np.pi, L) for key in ('AOA_az', 'DOA_az')}
    p.update({key: rng.uniform(-np.pi/2, np.pi/2, L) for key in ('AOA_el', 'DOA_el')})
    p['BETA'] = (rng.normal(size=L) + 1j*rng.normal(size=L))/np.sqrt(2*L)
    p['delay'] = rng.uniform(0, 1/fs, L)
    return p


def generate_FIM_deformed(NH, NV, dx, dy, pc, az, el, roll, xi):
    """Notebook geometry; vector xi follows nv-outer/nh-inner element order.

    For a (NV,NH) matrix use row-major flattening to match those coordinates.
    This fixes the notebook's xi.T.reshape(-1), inconsistent with its loop.
    The default codebook uses vectors, so this fix does not change that case.
    """
    xi = np.asarray(xi, float)
    require(xi.shape in ((NH*NV,), (NV, NH)), 'Invalid xi shape')
    xi = xi.ravel()
    ca, sa, ce, se, cr, sr = np.cos(az), np.sin(az), np.cos(el), np.sin(el), np.cos(roll), np.sin(roll)
    R = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]]) @ np.array([[ce,0,se],[0,1,0],[-se,0,ce]]) @ np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    x, y = np.meshgrid((np.arange(NH)-(NH-1)/2)*dx, (np.arange(NV)-(NV-1)/2)*dy)
    P = np.asarray(pc) + x.ravel()[:,None]*R[:,0] + y.ravel()[:,None]*R[:,1]
    return P+xi[:,None]*R[:,2], P, xi, R[:,0], R[:,1], R[:,2]


def steering_vector_fim(p0, zeta, wavelength, phi, theta):
    """Explicit (3,N) inputs avoid ambiguous 3x3 coordinate inference."""
    require(p0.ndim == 2 and p0.shape[0] == 3 and zeta.shape == p0.shape, 'Geometry must be (3,N)')
    u = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return np.exp(2j*np.pi/wavelength*(u @ (p0+zeta)))/np.sqrt(p0.shape[1])


def path_atoms(params, P_B0, Zeta_B, P_U0, Zeta_U, wavelength, fs, K):
    params = {key: np.asarray(value).ravel() for key,value in params.items()}
    atoms = []
    for l in range(len(params['BETA'])):
        b = steering_vector_fim(P_B0, Zeta_B, wavelength, params['AOA_az'][l], params['AOA_el'][l])
        u = steering_vector_fim(P_U0, Zeta_U, wavelength, params['DOA_az'][l], params['DOA_el'][l])
        phase = np.exp(-2j*np.pi*np.arange(K)/K*fs*params['delay'][l])
        atoms.append(b[:,None,None]*u.conj()[None,:,None]*phase[None,None,:])
    return np.stack(atoms, axis=-1)


def build_H_fim_from_paths(params, P_B0, Zeta_B, P_U0, Zeta_U, wavelength, fs, K):
    return path_atoms(params, P_B0, Zeta_B, P_U0, Zeta_U, wavelength, fs, K) @ np.asarray(params['BETA']).ravel()


def generate_multi_pilots(N_U, Tpilots, rng):
    n = np.arange(N_U)
    F = np.exp(-2j*np.pi*n[:,None]*n[None,:]/N_U)/np.sqrt(N_U)
    return F[:,:,None]*np.exp(2j*np.pi*rng.random((N_U,Tpilots)))[None,:,:]


def received(H, S):
    return np.einsum('buk,uvt->bvkt', H, S, optimize=True)


def pilot_transmission_multi(H, S_all, SNR_dB, rng):
    clean = received(H, S_all)
    sigma2 = float(np.mean(abs(clean)**2)/10**(SNR_dB/10))
    require(np.isfinite(sigma2) and sigma2 > 0, 'Invalid noise variance')
    noise = np.sqrt(sigma2/2)*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))
    return clean+noise, sigma2, clean


def make_codebook(a, rng):
    wave = 3e8/a.fc
    positions, bases = [], []
    for nh,nv,center,el in ((a.NH_B,a.NV_B,[0,0,0],np.pi/2), (a.NH_U,a.NV_U,[5*wave,0,0],-np.pi/2)):
        _, p, _, _, _, normal = generate_FIM_deformed(nh,nv,a.spacing*wave,a.spacing*wave,center,0,el,0,np.zeros(nh*nv))
        xi = rng.uniform(-1,1,(a.M,nh*nv))
        xi /= np.max(abs(xi),axis=1,keepdims=True)
        positions.append(p.T)
        bases.append(wave*xi[:,None,:]*normal[None,:,None])
    return positions[0], positions[1], bases[0], bases[1]


def pack_params(p, fs):
    L = len(p['BETA'])
    return np.column_stack((p['BETA'].real*np.sqrt(L),p['BETA'].imag*np.sqrt(L),fs*p['delay'],p['AOA_az'],p['AOA_el'],p['DOA_az'],p['DOA_el'])).ravel()


def unpack_params(q, fs):
    q = q.reshape(-1,7)
    return dict(BETA=(q[:,0]+1j*q[:,1])/np.sqrt(len(q)),delay=q[:,2]/fs,AOA_az=q[:,3],AOA_el=q[:,4],DOA_az=q[:,5],DOA_el=q[:,6])


def derivative(fun, q, step):
    cols = []
    for j in range(len(q)):
        delta = np.zeros_like(q)
        delta[j] = step
        cols.append((fun(q+delta)-fun(q-delta))/(2*step))
    return np.column_stack(cols)


def local_diagnostics(p, channel, S, variances, H0, a):
    q = pack_params(p,a.fs)
    def obs(qv):
        ps = unpack_params(qv,a.fs)
        return realvec(np.concatenate([received(channel(ps,m),S).ravel()/np.sqrt(variances[m]/2) for m in range(a.M)]))
    def target(qv):
        return realvec(channel(unpack_params(qv,a.fs),-1))
    D = derivative(obs,q,a.fd_step)
    Dhalf = derivative(obs,q,a.fd_step/2)
    convergence = np.linalg.norm(D-Dhalf)/max(np.linalg.norm(Dhalf),np.finfo(float).tiny)
    require(convergence < 1e-3, 'Jacobian finite differences did not converge; decrease --fd-step')
    T = derivative(target,q,a.fd_step/2)
    _, s, vh = np.linalg.svd(Dhalf,full_matrices=False)
    rank = int(np.sum(s > a.rcond*s[0]))
    null = vh[rank:].T
    null_effect = float(np.linalg.norm(T@null)/max(np.linalg.norm(T),np.finfo(float).tiny))
    identifiable = null_effect < a.null_tol
    crb = float(np.sum((T@vh[:rank].T/s[:rank])**2)/energy(H0)) if identifiable else None
    return dict(jacobian_rank=rank,parameter_count=len(q),jacobian_condition=float(s[0]/s[-1]) if rank==len(q) else None,
                target_null_fraction=null_effect,target_locally_estimable=identifiable,target_crb_nmse=crb,fd_relative_change=float(convergence))


def oracle_diagnostics(p, atoms, S, Y, variances, H0, a):
    A0 = atoms(-1).reshape(-1,a.L)
    blocks = []
    for m in range(a.M):
        Am = atoms(m)
        blocks.append(np.column_stack([received(Am[...,l],S).ravel() for l in range(a.L)]))
    A = np.vstack([b/np.sqrt(v) for b,v in zip(blocks,variances)])
    y = np.concatenate([z.ravel()/np.sqrt(v) for z,v in zip(Y,variances)])
    beta, _, rank, s = np.linalg.lstsq(A,y,rcond=a.rcond)
    clean_beta = np.linalg.lstsq(A,A@p['BETA'],rcond=a.rcond)[0]
    return dict(gain_oracle_rank=int(rank),gain_oracle_condition=float(s[0]/s[-1]) if rank==a.L else None,
                gain_oracle_nmse=nmse((A0@beta).reshape(H0.shape),H0),
                gain_oracle_noiseless_nmse=nmse((A0@clean_beta).reshape(H0.shape),H0))


def neighbors(X, H, rng):
    """Exact neighbors, chunked Gram distances; no N x N x feature allocation."""
    norms = np.sum(abs(X)**2,axis=1)
    distances, target_distances, random_distances = [], [], []
    for start in range(0,len(X),16):
        stop = min(start+16,len(X))
        d = np.maximum(norms[start:stop,None]+norms[None,:]-2*(X[start:stop]@X.conj().T).real,0)
        d /= np.maximum((norms[start:stop,None]+norms[None,:])/2,np.finfo(float).tiny)
        d[np.arange(stop-start),np.arange(start,stop)] = np.inf
        js = np.argmin(d,axis=1)
        for row,j in enumerate(js):
            i = start+row
            r = (i+int(rng.integers(1,len(X))))%len(X)
            distances.append(float(np.sqrt(d[row,j])))
            target_distances.append(energy(H[i]-H[j])/((energy(H[i])+energy(H[j]))/2))
            random_distances.append(energy(H[i]-H[r])/((energy(H[i])+energy(H[r]))/2))
    return dict(input_distance=distances,target_squared_distance=target_distances,random_target_squared_distance=random_distances)


def sanity(a, PB, PU, ZB, ZU, S, p):
    wave = 3e8/a.fc
    require(all(np.isfinite(x).all() for x in (PB,PU,ZB,ZU,S)), 'Nonfinite geometry/pilots')
    for t in range(a.Tpilots):
        require(np.allclose(S[:,:,t]@S[:,:,t].conj().T,np.eye(PU.shape[1]),atol=1e-12), 'Pilots not unitary')
    H = build_H_fim_from_paths(p,PB,np.zeros_like(PB),PU,np.zeros_like(PU),wave,a.fs,a.K)
    require(H.shape==(PB.shape[1],PU.shape[1],a.K) and np.isfinite(H).all() and energy(H)>0,'Invalid channel')
    require(np.linalg.matrix_rank(H[:,:,0])<=a.L,'Channel rank exceeds path count')
    for t in range(a.Tpilots):
        recovered = np.einsum('bvk,vu->buk',received(H,S)[...,t],S[:,:,t].conj().T)
        require(nmse(recovered,H)<1e-24,'Noiseless pilot inversion failed')
    perm = np.arange(a.L)[::-1]
    Hp = build_H_fim_from_paths({k:v[perm] for k,v in p.items()},PB,np.zeros_like(PB),PU,np.zeros_like(PU),wave,a.fs,a.K)
    require(nmse(Hp,H)<1e-24,'Path permutation invariance failed')
    for position in (PB,PU):
        steering = steering_vector_fim(position,np.zeros_like(position),wave,0.4,0.7)
        require(np.isclose(energy(steering),1),'Steering normalization failed')
    return ['finite geometry/pilots','unitary pilots','shape/energy/rank','noiseless pilot inversion','path permutation','unit-norm steering']


def run(a):
    streams = np.random.SeedSequence(a.seed).spawn(3)
    geom_rng, path_rng = (np.random.default_rng(s) for s in streams[:2])
    PB,PU,ZB,ZU = make_codebook(a,geom_rng)
    S = generate_multi_pilots(PU.shape[1],a.Tpilots,geom_rng)
    paths = [generate_path_parameters(a.L,a.fc,a.fs,path_rng) for _ in range(a.samples)]
    checks = sanity(a,PB,PU,ZB,ZU,S,paths[0])
    output = Path(a.output)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'codebook.npz',P_B0=PB,P_U0=PU,Zeta_B_base=ZB,Zeta_U_base=ZU,S_all=S)
    summaries, details = [], []
    for alpha in a.alphas:
        rng = np.random.default_rng(streams[2])  # paired noise across alpha
        def atoms(p,m):
            zb,zu = (np.zeros_like(PB),np.zeros_like(PU)) if m<0 else (alpha*ZB[m],alpha*ZU[m])
            return path_atoms(p,PB,zb,PU,zu,3e8/a.fc,a.fs,a.K)[:,:,tones,:]
        def channel(p,m):
            return atoms(p,m)@p['BETA']
        tones = slice(None) if a.pair_start is None else slice(a.pair_start,a.pair_start+2)
        Xclean, Xnoisy, targets, rows = [], [], [], []
        model_inputs, model_targets, model_scales = [], [], []
        noise_sum, expected_noise_sum, noise_power_variance = 0.,0.,0.
        for i,p in enumerate(paths):
            H0 = channel(p,-1)
            Hs = [channel(p,m) for m in range(a.M)]
            require(all(np.isfinite(h).all() and energy(h)>0 for h in Hs) and energy(H0)>0,'Nonfinite or zero channel')
            if alpha==0:
                require(all(nmse(h,H0)<1e-24 for h in Hs),'Zero deformation identity failed')
            # Generate full-band noise first: pair and full-band experiments
            # have identical draws/variance for matching seeds and samples.
            full_Hs = Hs if a.pair_start is None else [
                build_H_fim_from_paths(p,PB,alpha*ZB[m],PU,alpha*ZU[m],3e8/a.fc,a.fs,a.K)
                for m in range(a.M)]
            full_tuples = [pilot_transmission_multi(h,S,a.snr_db,rng) for h in full_Hs]
            tuples = [(y[:,:,tones,:],v,c[:,:,tones,:]) for y,v,c in full_tuples]
            Y,vs,clean = map(list,zip(*tuples))
            for y,c,v in zip(Y,clean,vs):
                noise_sum += energy(y-c)
                expected_noise_sum += v*y.size
                noise_power_variance += v*v*y.size
            rho = [min(1.,float(abs(np.vdot(h,H0))/np.sqrt(energy(h)*energy(H0)))) for h in Hs]
            V = np.stack([h.ravel()/np.sqrt(energy(h)) for h in Hs])
            eig = np.maximum(np.linalg.eigvalsh(V@V.conj().T),0)
            deprojected = [np.mean(np.stack([np.einsum('bvk,vu->buk',y[...,t],S[:,:,t].conj().T) for t in range(a.Tpilots)]),axis=0) for y in Y]
            row = dict(sample=i,rho_mean=float(np.mean(rho)),residual_nmse=float(np.mean([nmse(h,H0) for h in Hs])),energy_ratio=float(np.mean([energy(h)/energy(H0) for h in Hs])),view_effective_rank=float(eig.sum()**2/np.sum(eig**2)),first_view_nmse=nmse(deprojected[0],H0),mean_view_nmse=nmse(np.mean(deprojected,axis=0),H0))
            if i<a.diagnostic_samples:
                row.update(oracle_diagnostics(p,lambda m:atoms(p,m),S,Y,vs,H0,a))
                row.update(local_diagnostics(p,channel,S,vs,H0,a))
                if (i+1)%10==0:
                    print(f'  alpha={alpha:g}: local diagnostics {i+1}/{a.diagnostic_samples}',flush=True)
            row.update(describe(Hs, H0, deprojected, vs))
            if getattr(a, 'export_inputs', False):
                require(a.Tpilots == 1 and a.pair_start is not None, 'Input export requires one pilot and a tone pair')
                R = np.asarray([x/(1+v) for x,v in zip(deprojected,vs)])
                scale = row['input_scale']
                packed = np.stack([part for r in R for part in (r[:,:,0].real,r[:,:,0].imag,r[:,:,1].real,r[:,:,1].imag)]) / scale
                target = np.stack([H0[:,:,0].real,H0[:,:,0].imag,H0[:,:,1].real,H0[:,:,1].imag]) / scale
                model_inputs.append(packed.astype(np.float32))
                model_targets.append(target.astype(np.float32))
                model_scales.append(scale)
            rows.append(row)
            Xclean.append(np.concatenate([c.ravel() for c in clean]))
            Xnoisy.append(np.concatenate([y.ravel() for y in Y]))
            targets.append(H0.ravel())
        if model_inputs:
            np.savez_compressed(output/f'inputs_alpha_{alpha:g}.npz', X=np.asarray(model_inputs),
                                H_target=np.asarray(model_targets), scale=np.asarray(model_scales),
                                sample=np.arange(a.samples), alpha=alpha, seed=a.seed, pair_start=a.pair_start,
                                path_count=a.L, snr_db=a.snr_db)
        ratio = noise_sum/expected_noise_sum
        # Weighted sum of exponential noise powers: conservative 8-sigma bound.
        require(abs(ratio-1)<8*np.sqrt(noise_power_variance)/expected_noise_sum,'Empirical noise power inconsistent with specified variance')
        nn_clean = neighbors(np.asarray(Xclean),np.asarray(targets),rng)
        nn_noisy = neighbors(np.asarray(Xnoisy),np.asarray(targets),rng)
        summary = dict(alpha=alpha,pair_start=a.pair_start,observed_tones=a.K if a.pair_start is None else 2,diagnostic_samples=a.diagnostic_samples,empirical_noise_ratio=ratio)
        for key in rows[0]:
            if key=='sample':
                continue
            values = [r[key] for r in rows if key in r and r[key] is not None]
            summary[key+'_median'] = float(np.median(values)) if values else None
            if key in ('gain_oracle_nmse','target_crb_nmse','jacobian_condition'):
                summary[key+'_p90'] = float(np.quantile(values,.9)) if values else None
                summary[key+'_missing_count'] = a.diagnostic_samples-len(values)
        for label,nn in [('clean',nn_clean),('noisy',nn_noisy)]:
            for key,values in nn.items():
                summary[label+'_nn_'+key+'_median'] = float(np.median(values))
                summary[label+'_nn_'+key+'_p10'] = float(np.quantile(values,0.1))
        summary['local_target_failure_count'] = sum(r.get('target_locally_estimable') is False for r in rows)
        summaries.append(summary)
        details.append(dict(alpha=alpha,samples=rows,nearest_neighbors_clean=nn_clean,nearest_neighbors_noisy=nn_noisy))
        print(f"b/lambda={alpha:g}: rho={summary['rho_mean_median']:.3f}, mean-view NMSE={10*np.log10(max(summary['mean_view_nmse_median'],1e-300)):.2f} dB, local target failures={summary['local_target_failure_count']}/{a.diagnostic_samples}",flush=True)
    report = dict(configuration=vars(a),sanity_checks=checks,summary=summaries,details=details,
        interpretation='Local/oracle evidence only. Null conditions and CRB depend on numerical tolerance. Observations and targets both use the configured tones, with full-band noise calibration. No CNN trained. Additional views add pilot energy. Medians/p90 exclude missing values; inspect missing and failure counts. Read module docstring.')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    with (output/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    if a.plot:
        try:
            import matplotlib
        except ImportError:
            print('Plots skipped: install matplotlib with: python -m pip install matplotlib')
            print(f'JSON, CSV and codebook saved to {output.resolve()}')
            return
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,3,figsize=(14,4))
        for ax,key,label in zip(axs,['rho_mean_median','mean_view_nmse_median','target_null_fraction_median'],['Channel correlation','Mean-view NMSE (linear)','Target null-space fraction']):
            ax.plot(a.alphas,[s[key] for s in summaries],'o-')
            ax.set(xlabel='b / wavelength',ylabel=label)
            ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(output/'diagnostics.png',dpi=160)
        plt.close(fig)
    print(f"Saved results to {output.resolve()}")


def parse_args(argv=None):
    """Parse explicit options, or command-line options when argv is None.

    In a notebook use run(parse_args(['--quick'])) or run(parse_args([])).
    Explicit lists keep Jupyter's kernel arguments out of this parser.
    """
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for name,default in [('NH_B',5),('NV_B',5),('NH_U',5),('NV_U',5),('K',32),('M',8),('Tpilots',1),('L',3),('samples',100),('diagnostic-samples',3),('seed',1)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('fc',28e9),('fs',100e3),('spacing',0.125),('snr-db',10.),('fd-step',1e-5),('rcond',1e-8),('null-tol',1e-6)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--alphas',nargs='+',type=float,default=[0,.05,.1,.2,.3,.5])
    p.add_argument('--output',default='estimability_results')
    p.add_argument('--plot',action='store_true')
    p.add_argument('--pair-start',type=int,default=None,help='Zero-based k: use only subcarriers k,k+1, keeping original K. Default: full band.')
    p.add_argument('--quick',action='store_true',help='8 samples, one local/oracle diagnostic per alpha; physical dimensions unchanged')
    a=p.parse_args(argv)
    if a.quick:
        a.samples=8
        a.diagnostic_samples=1
        a.alphas=[0.,.2,.5]
    require(all(getattr(a,k)>0 for k in ['NH_B','NV_B','NH_U','NV_U','K','M','Tpilots','L','fc','fs','spacing','fd_step','rcond','null_tol']), 'Dimensions/scales must be positive')
    require(a.seed>=0 and a.samples>=3 and 1<=a.diagnostic_samples<=a.samples,'Need samples>=3 and 1<=diagnostic-samples<=samples; seed>=0')
    require(all(np.isfinite(v) and v>=0 for v in a.alphas),'Invalid morphing range')
    require(all(np.isfinite(getattr(a,k)) for k in ['fc','fs','spacing','snr_db','fd_step','rcond','null_tol']), 'Nonfinite option')
    require(0<a.rcond<1 and 0<a.null_tol<1,'Tolerances must lie in (0,1)')
    require(a.pair_start is None or 0<=a.pair_start<a.K-1,'pair-start must be between 0 and K-2')
    observed_tones = a.K if a.pair_start is None else 2
    require(2*a.M*a.NH_B*a.NV_B*a.NH_U*a.NV_U*observed_tones*a.Tpilots>=7*a.L,'Too few real measurements for this reduced-SVD diagnostic; increase dimensions/views')
    return a


if __name__=='__main__':
    # Pasted notebook cells execute as __main__ too. Kernel launcher flags
    # belong to Jupyter, not this experiment. CLI parsing remains strict.
    run(parse_args([] if 'ipykernel' in sys.modules else None))
