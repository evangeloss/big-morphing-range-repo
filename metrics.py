"""Additional descriptive metrics; no learned model or universal thresholds."""
import numpy as np


def describe(hs, target, estimates, variances):
    tiny = np.finfo(float).tiny
    e = max(float(np.vdot(target, target).real), tiny)
    h = np.asarray(hs)
    r = np.asarray([x / (1 + v) for x, v in zip(estimates, variances)])
    v = h.reshape(len(h), -1)
    gram = v @ v.conj().T
    norms = np.sqrt(np.maximum(np.diag(gram).real, tiny))
    coherence = np.abs(gram) / (norms[:, None] * norms[None, :])
    off = coherence[np.triu_indices(len(h), 1)]
    product = h * target.conj()[None]
    weights = np.abs(product)
    phase = np.angle(product)
    scale = max(float(np.max(np.abs(r))), 1e-8)
    residual = target - r[0]
    return {
        'view_coherence_mean': float(off.mean()) if len(off) else 1.,
        'phase_weighted_rms_rad': float(np.sqrt(np.sum(weights * phase**2) / max(weights.sum(), tiny))),
        'magnitude_difference_nmse': float(np.mean(np.sum((np.abs(h)-np.abs(target))**2, axis=(1,2,3))) / e),
        'ridge_first_view_nmse': float(np.sum(np.abs(residual)**2) / e),
        'ridge_mean_view_nmse': float(np.sum(np.abs(r.mean(axis=0)-target)**2) / e),
        'input_scale': scale,
        'normalized_residual_rms': float(np.sqrt(np.mean(np.abs(residual/scale)**2))),
        'normalized_target_peak': float(np.max(np.abs(target))/scale),
        'target_energy': e,
    }
