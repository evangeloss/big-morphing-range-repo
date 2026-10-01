"""Run oracle/error-floor diagnostics using an existing trained hybrid checkpoint."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from simulator import Simulator
from bridge import PhysicsBridge
from model import Reconstructor
from train import load_state

from oracle_error_floor import (
    evaluate_snr_sweep,
)


def write_csv(path, rows):

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help=(
            "Existing train.py results folder "
            "containing configuration.json, "
            "bridge_large.pt and checkpoint."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "oracle_floor_results"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Trained model seed to evaluate.",
    )

    parser.add_argument(
        "--regime",
        choices=[
            "large",
            "small",
        ],
        default="large",
    )

    parser.add_argument(
        "--scenes",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--snrs",
        nargs="+",
        type=float,
        default=[
            -10,
            -5,
            0,
            5,
            10,
            15,
            20,
        ],
    )

    parser.add_argument(
        "--scene-seed",
        type=int,
        default=710000000,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
    )

    args = parser.parse_args()

    args.results = (
        args.results
        .resolve()
    )

    args.output = (
        args.output
        .resolve()
    )

    if args.quick:
        args.scenes = 16

        args.batch_size = min(
            args.batch_size,
            8,
        )

    if (
        args.scenes <= 0
        or args.batch_size <= 0
        or not args.snrs
        or any(
            not math.isfinite(x)
            for x in args.snrs
        )
    ):
        parser.error(
            "Use positive scenes/batch size "
            "and finite SNR values."
        )

    configuration_file = (
        args.results
        / "configuration.json"
    )

    if not configuration_file.is_file():
        parser.error(
            f"configuration.json not found in "
            f"{args.results}"
        )

    config = json.loads(
        configuration_file.read_text()
    )

    alpha = float(
        config[
            f"{args.regime}_alpha"
        ]
    )

    width = int(
        config["width"]
    )

    geometry_seed = int(
        config["geometry_seed"]
    )

    paths = int(
        config["paths"]
    )

    pair_start = int(
        config["pair_start"]
    )

    bridge_file = (
        args.results
        / f"bridge_{args.regime}.pt"
    )

    checkpoint = (
        args.results
        / f"seed_{args.seed}"
        / f"{args.regime}_hybrid"
        / "best.pt"
    )

    geometry_file = (
        args.results
        / "geometry.pt"
    )

    for file in (
        bridge_file,
        checkpoint,
        geometry_file,
    ):

        if not file.is_file():
            parser.error(
                f"Required file not found: "
                f"{file}"
            )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    torch.set_num_threads(
        min(
            4,
            torch.get_num_threads(),
        )
    )

    print(
        "Device:",
        device,
        flush=True,
    )

    print(
        f"Results: {args.results}\n"
        f"Regime: {args.regime}\n"
        f"b/lambda: {alpha:g}\n"
        f"Checkpoint: {checkpoint}",
        flush=True,
    )

    # =========================================================
    # Recreate exact simulator
    # =========================================================

    sim = Simulator(
        geometry_seed,
        paths,
        pair_start,
    ).to(device)

    sim.load_state_dict(
        torch.load(
            geometry_file,
            map_location=device,
            weights_only=True,
        )
    )

    # =========================================================
    # Load the exact physics bridge
    # =========================================================

    bridge_state = torch.load(
        bridge_file,
        map_location=device,
        weights_only=True,
    )

    bridge = PhysicsBridge(
        bridge_state["U"],
        bridge_state["T"],
        bridge_state[
            "eigenvalues"
        ],
    ).to(device)

    # =========================================================
    # Load EXISTING hybrid model
    # =========================================================

    model = Reconstructor(
        bridge,
        hybrid=True,
        width=width,
    ).to(device)

    checkpoint_state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )

    load_state(
        model,
        checkpoint_state["model"],
    )

    model.eval()
    bridge.eval()

    # =========================================================
    # Oracle experiment
    # =========================================================

    rows = evaluate_snr_sweep(
        sim=sim,
        bridge=bridge,
        model=model,
        alpha=alpha,
        snrs=args.snrs,
        scenes=args.scenes,
        batch_size=args.batch_size,
        seed=args.scene_seed,
    )

    # =========================================================
    # Save numerical results
    # =========================================================

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        args.output
        / "oracle_floor.csv",
        rows,
    )

    metadata = {
        "results":
            str(args.results),

        "regime":
            args.regime,

        "alpha":
            alpha,

        "model_seed":
            args.seed,

        "scene_seed":
            args.scene_seed,

        "scenes_per_snr":
            args.scenes,

        "batch_size":
            args.batch_size,

        "snrs":
            [
                float(x)
                for x in args.snrs
            ],

        "device":
            str(device),

        "checkpoint_epoch":
            checkpoint_state.get(
                "epoch"
            ),

        "checkpoint_validation_nmse":
            checkpoint_state.get(
                "validation_nmse"
            ),

        "bridge_info":
            bridge_state.get(
                "info",
                {},
            ),
    }

    (
        args.output
        / "oracle_floor_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    # =========================================================
    # Main NMSE plot
    # =========================================================

    x = np.asarray([
        row["snr_db"]
        for row in rows
    ])

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.plot(
        x,
        [
            row["genie_oracle_db"]
            for row in rows
        ],
        marker="o",
        label="Genie structural oracle",
    )

    ax.plot(
        x,
        [
            row[
                "physics_noiseless_db"
            ]
            for row in rows
        ],
        marker="s",
        linestyle="--",
        label="Noiseless physics floor",
    )

    ax.plot(
        x,
        [
            row["physics_db"]
            for row in rows
        ],
        marker="^",
        label="Physics",
    )

    ax.plot(
        x,
        [
            row["hybrid_db"]
            for row in rows
        ],
        marker="D",
        label="Hybrid",
    )

    ax.set(
        xlabel="SNR (dB)",
        ylabel="NMSE (dB)",
        title=(
            "Oracle error-floor analysis, "
            f"b/lambda={alpha:g}"
        ),
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    for extension in (
        "png",
        "pdf",
    ):

        fig.savefig(
            args.output
            / (
                "oracle_floor."
                + extension
            ),
            dpi=200,
        )

    plt.close(fig)

    # =========================================================
    # Recovery-fraction plot
    # =========================================================

    fig, ax = plt.subplots(
        figsize=(8.5, 4.8)
    )

    recovery = (
        100
        * np.asarray([
            row[
                "recoverable_gap_removed"
            ]
            for row in rows
        ])
    )

    ax.plot(
        x,
        recovery,
        marker="o",
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        100,
        linestyle="--",
        linewidth=1,
    )

    ax.set(
        xlabel="SNR (dB)",
        ylabel=(
            "Recoverable physics "
            "error removed (%)"
        ),
        title=(
            "Hybrid correction "
            "relative to genie floor"
        ),
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    for extension in (
        "png",
        "pdf",
    ):

        fig.savefig(
            args.output
            / (
                "oracle_recovery."
                + extension
            ),
            dpi=200,
        )

    plt.close(fig)

    print(
        "\nPerfect-oracle sanity "
        "check (worst dB):",
        max(
            row[
                "perfect_oracle_db"
            ]
            for row in rows
        ),
    )

    print(
        "Saved:",
        args.output
        / "oracle_floor.csv",
    )

    print(
        "Saved:",
        args.output
        / "oracle_floor.png",
    )

    print(
        "Saved:",
        args.output
        / "oracle_recovery.png",
    )


if __name__ == "__main__":
    main()