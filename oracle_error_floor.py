"""Oracle/error-floor evaluation for the fixed-amplitude hybrid FIM estimator.

This module uses the exact latent channel parameters from ``Simulator`` only for
oracle diagnostics. The trained hybrid and physics baselines receive exactly the
same quantities as in normal evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from simulator import pack, unpack


@dataclass
class OracleBatch:
    perfect: torch.Tensor
    genie: torch.Tensor
    physics_noiseless: torch.Tensor
    physics: torch.Tensor
    hybrid: torch.Tensor
    condition_number: torch.Tensor


def complex_nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Per-scene NMSE for complex tensors.

    Batch dimension must be first.
    """
    dims = tuple(range(1, pred.ndim))

    numerator = (
        (pred - target)
        .abs()
        .square()
        .sum(dims)
    )

    denominator = (
        target
        .abs()
        .square()
        .sum(dims)
        .clamp_min(1e-12)
    )

    return numerator / denominator


def mean_db(values) -> float:
    """
    Convert mean linear NMSE to dB.

    This follows the convention already used in train.py:
        mean linear NMSE -> 10 log10(.)
    """
    value = float(np.mean(values))

    return 10.0 * math.log10(
        max(value, 1e-15)
    )


def _unit_path_components(
    sim,
    db,
    du,
    delay,
    alpha,
    rigid=False,
):
    """
    Construct the contribution of every unit-gain path.

    Returns
    -------
    components :
        [B, M, 25, 25, 2, L]

    For rigid=True:
        M = 1.

    Importantly, this calls sim.atoms(), so the oracle uses
    exactly the same channel model as the simulator.
    """

    atoms = sim.atoms(
        db,
        du,
        alpha,
        rigid=rigid,
    )
    # [B,M,25,25,L]

    tone_index = torch.arange(
        sim.pair,
        sim.pair + 2,
        device=db.device,
        dtype=delay.dtype,
    )

    phase = torch.exp(
        -2j
        * math.pi
        * tone_index[None, :, None]
        / 32
        * delay[:, None, :]
    )
    # [B,2,L]

    return (
        atoms.unsqueeze(-2)
        * phase[:, None, None, None, :, :]
    )
    # [B,M,25,25,2,L]


@torch.no_grad()
def genie_gain_oracle(
    sim,
    data,
    alpha,
):
    """
    Structural oracle.

    The oracle KNOWS:
        - true AoA/AoD directions
        - true delays
        - exact FIM geometry
        - number of paths L
        - noise variance

    The oracle DOES NOT KNOW:
        - complex gains beta_l

    beta is estimated from the same noisy observations used
    by the normal estimators.

    Weighted LS / GLS is used because each deformation view
    has a known noise variance.
    """

    components = _unit_path_components(
        sim,
        data["db"],
        data["du"],
        data["delay"],
        alpha,
        rigid=False,
    )

    B, M, I, J, K, L = components.shape

    # Measurement dictionary:
    #
    # y = A beta + n
    #
    A = components.reshape(
        B,
        M * I * J * K,
        L,
    )

    y = data["observed"].reshape(
        B,
        M * I * J * K,
    )

    # ---------------------------------------------------------
    # Whitening using the exact per-view noise variance
    # ---------------------------------------------------------

    sigma = (
        torch.sqrt(data["variance"])
        .clamp_min(1e-12)
    )
    # [B,M]

    weights = (
        (1.0 / sigma)
        [:, :, None, None, None]
        .expand(B, M, I, J, K)
        .reshape(B, -1)
    )

    Aw = A * weights[..., None]
    yw = y * weights

    # ---------------------------------------------------------
    # Estimate beta
    # ---------------------------------------------------------

    AH = Aw.conj().transpose(-2, -1)

    gram = AH @ Aw
    rhs = AH @ yw[..., None]

    # L is only 3 by default.
    # pinv makes rare nearly-singular scenes safe.
    beta_hat = (
        torch.linalg.pinv(
            gram,
            hermitian=True,
        )
        @ rhs
    ).squeeze(-1)

    # ---------------------------------------------------------
    # Reconstruct target rigid / undeformed channel
    # ---------------------------------------------------------

    rigid_components = _unit_path_components(
        sim,
        data["db"],
        data["du"],
        data["delay"],
        alpha,
        rigid=True,
    )

    H_hat = torch.einsum(
        "bmijkl,bl->bmijk",
        rigid_components,
        beta_hat,
    )[:, 0]

    # ---------------------------------------------------------
    # Conditioning diagnostic
    # ---------------------------------------------------------

    eig = (
        torch.linalg
        .eigvalsh(gram)
        .real
        .clamp_min(0)
    )

    condition = torch.sqrt(
        eig[:, -1]
        /
        eig[:, 0].clamp_min(
            eig[:, -1] * 1e-12
            + 1e-30
        )
    )

    return H_hat, condition


@torch.no_grad()
def evaluate_batch(
    sim,
    bridge,
    model,
    data,
    alpha,
):
    """
    Evaluate every curve on exactly the same channel realization.
    """

    # True rigid target in physical complex units.
    target = data["target"][:, 0]
    # [B,25,25,2]

    # =========================================================
    # 1. Perfect parameter oracle
    # =========================================================
    #
    # This knows beta too.
    # It is ONLY a reconstruction sanity check.
    #

    rigid_components = _unit_path_components(
        sim,
        data["db"],
        data["du"],
        data["delay"],
        alpha,
        rigid=True,
    )

    perfect_hat = torch.einsum(
        "bmijkl,bl->bmijk",
        rigid_components,
        data["beta"],
    )[:, 0]

    # =========================================================
    # 2. Genie structural oracle
    # =========================================================

    genie_hat, condition = genie_gain_oracle(
        sim,
        data,
        alpha,
    )

    # =========================================================
    # 3. Normal physics estimator
    # =========================================================
    #
    # EXACT same call already used in your model.
    #

    physics_norm = bridge(
        data["x"],
        data["variance"],
    )

    physics_hat = (
        unpack(physics_norm)[:, 0]
        * data["scale"][:, None, None, None]
    )

    # =========================================================
    # 4. Hybrid model
    # =========================================================
    #
    # Existing checkpoint, no retraining.
    #

    hybrid_norm = model(
        data["x"],
        data["variance"],
        data["scale"],
    )

    hybrid_hat = (
        unpack(hybrid_norm)[:, 0]
        * data["scale"][:, None, None, None]
    )

    # =========================================================
    # 5. Noiseless physics floor
    # =========================================================
    #
    # Same bridge.
    # Same channel realization.
    # But remove AWGN entirely.
    #

    clean_scale = (
        data["clean"]
        .abs()
        .flatten(1)
        .amax(1)
        .clamp_min(1e-8)
    )

    clean_x = (
        pack(data["clean"])
        / clean_scale[:, None, None, None]
    )

    zero_variance = torch.zeros_like(
        data["variance"]
    )

    physics_clean_norm = bridge(
        clean_x,
        zero_variance,
    )

    physics_clean_hat = (
        unpack(physics_clean_norm)[:, 0]
        * clean_scale[:, None, None, None]
    )

    return OracleBatch(
        perfect=complex_nmse(
            perfect_hat,
            target,
        ),

        genie=complex_nmse(
            genie_hat,
            target,
        ),

        physics_noiseless=complex_nmse(
            physics_clean_hat,
            target,
        ),

        physics=complex_nmse(
            physics_hat,
            target,
        ),

        hybrid=complex_nmse(
            hybrid_hat,
            target,
        ),

        condition_number=condition,
    )


@torch.no_grad()
def evaluate_snr_sweep(
    sim,
    bridge,
    model,
    alpha,
    snrs,
    scenes,
    batch_size,
    seed=710000000,
):
    """
    Evaluate all curves across SNR.

    Important:
    every SNR uses the same latent channel scenes.

    Therefore changes across SNR are paired comparisons,
    rather than Monte-Carlo differences caused by new channels.
    """

    model.eval()
    bridge.eval()

    rows = []

    for snr in snrs:

        store = {
            key: []
            for key in (
                "perfect",
                "genie",
                "physics_noiseless",
                "physics",
                "hybrid",
                "condition_number",
            )
        }

        for j, start in enumerate(
            range(
                0,
                scenes,
                batch_size,
            )
        ):

            n = min(
                batch_size,
                scenes - start,
            )

            # Same seed+j at every SNR.
            #
            # Simulator therefore generates identical:
            # db, du, beta, delay and standardized noise.
            #
            # Only the requested SNR changes.
            data = sim.batch(
                n,
                seed + j,
                alpha,
                snr,
                return_latents=True,
            )

            out = evaluate_batch(
                sim,
                bridge,
                model,
                data,
                alpha,
            )

            for key in store:
                values = (
                    getattr(out, key)
                    .detach()
                    .cpu()
                    .double()
                    .tolist()
                )

                store[key].extend(values)

        perfect = np.asarray(
            store["perfect"]
        )

        genie = np.asarray(
            store["genie"]
        )

        physics_clean = np.asarray(
            store["physics_noiseless"]
        )

        physics = np.asarray(
            store["physics"]
        )

        hybrid = np.asarray(
            store["hybrid"]
        )

        # -----------------------------------------------------
        # Fraction of theoretically recoverable physics error
        # removed by the NN.
        #
        # 0   = NN did nothing
        # 1   = reached genie floor
        # -----------------------------------------------------

        denominator = float(
            physics.mean()
            - genie.mean()
        )

        if abs(denominator) > 1e-15:
            recovery = float(
                (
                    physics.mean()
                    - hybrid.mean()
                )
                / denominator
            )
        else:
            recovery = float("nan")

        row = {
            "snr_db":
                float(snr),

            "perfect_oracle_nmse":
                float(perfect.mean()),

            "perfect_oracle_db":
                mean_db(perfect),

            "genie_oracle_nmse":
                float(genie.mean()),

            "genie_oracle_db":
                mean_db(genie),

            "physics_noiseless_nmse":
                float(
                    physics_clean.mean()
                ),

            "physics_noiseless_db":
                mean_db(
                    physics_clean
                ),

            "physics_nmse":
                float(
                    physics.mean()
                ),

            "physics_db":
                mean_db(
                    physics
                ),

            "hybrid_nmse":
                float(
                    hybrid.mean()
                ),

            "hybrid_db":
                mean_db(
                    hybrid
                ),

            "hybrid_gain_db":
                (
                    mean_db(physics)
                    - mean_db(hybrid)
                ),

            "recoverable_gap_removed":
                recovery,

            "oracle_condition_median":
                float(
                    np.median(
                        store[
                            "condition_number"
                        ]
                    )
                ),

            "oracle_condition_p95":
                float(
                    np.percentile(
                        store[
                            "condition_number"
                        ],
                        95,
                    )
                ),
        }

        rows.append(row)

        print(
            f"SNR {snr:>5g} dB | "
            f"genie "
            f"{row['genie_oracle_db']:8.3f} dB | "
            f"clean-physics "
            f"{row['physics_noiseless_db']:8.3f} dB | "
            f"physics "
            f"{row['physics_db']:8.3f} dB | "
            f"hybrid "
            f"{row['hybrid_db']:8.3f} dB | "
            f"gap removed "
            f"{100 * recovery:6.1f}%",
            flush=True,
        )

    return rows