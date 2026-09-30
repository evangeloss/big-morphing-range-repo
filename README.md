# Large-morphing channel reconstruction

**Objective:** train at b/lambda=0.5 and approach matched small-deformation accuracy. This project is a new experiment, not a claim that the gap is already solved.

## Run in Kaggle

1. Extract this ZIP. Upload the extracted files to a public GitHub repository (root or one subfolder).
2. Import `Kaggle_Large_Morphing.ipynb` into Kaggle, or paste `kaggle_launcher.py` into one code cell.
3. Enable Internet and a GPU. Set REPO_URL to the repository URL.
4. Run. The launcher installs dependencies and checks the numerical implementation before training.
5. Send back **results_for_review.zip**. It includes the report, errors, learning curves in CSV form and configuration. Model and optimizer checkpoints remain in the printed results folder; save those separately if you need to resume in another session.

Default: **one seed (11), four models**, up to 100 epochs each. One seed is a first experiment, not a robust seed-averaged conclusion. Set SEEDS=[11,22,33] for three-seed replication. QUICK=True runs all stages with a tiny network/data/dictionary in two epochs; its scores are only a software check.

LARGE_ALPHA defaults to 0.5, SMALL_ALPHA to 0.02. Each model trains at its fixed amplitude; this is not mixed-amplitude training. Change LARGE_ALPHA to focus on another known large value. An amplitude interval or unseen deformation shape is a separate experiment.

## What changed and why

The previous large-only network still needed to learn a large phase-dependent transformation. Supplying geometry through FiLM gave only a modest benefit. Here geometry explicitly constructs a linear reconstruction of the undeformed channel BEFORE the learned correction.

1. **Physics transport:** a low-rank LMMSE estimator combines all eight deformed views and all spatial entries, using their known positions and the simulator's angular prior.
2. **Learned refinement:** a multiscale convolutional network predicts the remaining correction to that reconstructed channel. It also sees all raw deprojected views and normalized receiver noise power.
3. **Fresh training scenes:** each optimizer step samples new independent propagation scenes, avoiding a fixed 4096-scene training set. No scenes are selected using test errors.
4. **Longer optimization:** AdamW, validation-driven learning-rate reduction, up to 100 epochs, and optional stopping only after at least 50 epochs, 25 stagnant epochs and learning-rate reductions.
5. **Matched small reference:** retrain the same methods at 0.02; judge both absolute NMSE and the large-minus-small gap. A configurable 2 dB target is an operational definition of “close,” not a theoretical guarantee.

## The four training arms

| Arm | Deformation | Prediction |
|---|---|---|
| large_hybrid | 0.5 | Physics estimate + learned correction |
| large_direct | 0.5 | Direct normalized channel prediction |
| small_hybrid | 0.02 | Matched small-deformation hybrid |
| small_direct | 0.02 | Matched small-deformation direct model |

All arms use the same neural parameter count, multiscale architecture, scene streams and optimizer settings. Direct models get zero-filled physics channels and no residual skip, but receive the same observations and noise metadata. The hybrid adds fixed physics buffers/computation, which are NOT included as trainable parameters; this is an informed-estimator comparison, not equal total computational cost. Pure physics, first-view and mean-view baselines are evaluated too.

The network uses GroupNorm rather than BatchNorm to avoid reliance on a running population normalization. The final layer starts at zero: hybrid initialization equals the physics estimator, direct initialization equals zero. The initial model is eligible for validation selection, so refinement need not replace a stronger physics estimate. Best validation checkpoint is selected separately for each arm. Test data never selects checkpoints.

## What the physics estimator knows

Known: exact fixed BS/UE geometry, current deformation amplitude, angular distribution used by the simulator, and calibrated per-view receiver noise variance. Unknown: the test scene's angles, gains and delays. The estimator does not read target channels or those latent parameters.

For random angular quadrature columns, let D stack the deformed steering outer-products over 8 views and let D0 contain the corresponding undeformed steering outer-products. Both use the same quadrature directions and are scaled by 1/sqrt(Q). The estimator approximates

`H_hat = D0 D^H (D D^H + sigma_squared I)^(-1) observations`.

It uses an eigendecomposition of D^H D, retains up to 384 modes above a relative 1e-6 eigenvalue threshold, and applies the estimator efficiently in that basis. Default Q=1024 independent prior direction samples. Directions are numerical integration samples, not a search over or disclosure of actual test-scene paths. Dictionary seed 314159 is independent of scene streams. `bridge_diagnostics.json` records retained prior energy and rank. High retained energy does not guarantee adequate quadrature coverage or target reconstruction.

The phase prior preserves the original polar-angle convention: phi uniform [-pi,pi], theta uniform [-pi/2,pi/2], direction=(sin(theta)cos(phi),sin(theta)sin(phi),cos(theta)). Path gains have expected total power 1. This linear estimator exploits the second-order prior; it does not fit the actual three paths. Nonlinear inference remains the network's job.

**Approximation:** the bridge averages eight per-view noise variances to use a common white-noise regularizer. The simulator retains their differences. It is not exact LMMSE for heteroscedastic noise. Receiver noise calibration is assumed available, not estimated from just two tones. For real data, replace that input with a receiver calibration/estimator and re-evaluate sensitivity. These assumptions give this project information not necessarily supplied to earlier checkpoints, so compare the new matched arms first.

If the physics-only baseline disappoints, inspect dictionary rank/coverage and noise assumptions before increasing the neural network. A quadrature convergence check can use --atoms 2048 --rank 512 in a separate run; it is more expensive and not an automatic improvement.

References: [LMMSE lecture](https://lall.stanford.edu/engr207b/lectures/regression_and_learning_2011_03_03_01.pdf), [PyTorch validation-driven learning-rate scheduler](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).

## Measurement protocol

- Same 5x5 arrays, lambda/8 spacing, 8 views, 28 GHz convention, K=32, three paths and geometry seed 2026 as the EDA family.
- Observe only tones 0 and 1. Delays follow fs*tau uniform [0,1]. Full-band clean signal power is computed analytically from path Gram matrices across all 32 tones.
- With full unitary pilots, inversion preserves iid complex Gaussian noise. Generate noisy deprojected channels directly, with the same distribution as pilot transmission followed by inversion. Random draws are not bitwise identical to the prior NumPy scripts.
- Inputs are raw deprojected channels, without the old tiny ridge shrinkage. Normalize all views and both targets by the peak observed magnitude only. No target-assisted normalization.
- Each step has 32 fresh scenes; default 64 steps/epoch = 2048 scenes/epoch, at most 204800 training scenes per model. SNR cycles evenly through 0,10,20 dB. Training streams are paired across arms until any arm stops; actual optimization budgets may differ due to stopping.
- Train, validation and test use disjoint seed ranges. Validation is 256 fixed scenes with balanced SNR; test is 300 independent scenes per seed, reused in paired comparisons across amplitudes and SNRs. These repeats are not counted as additional independent scenes. Do not change batch size between matched tests; batch seeding is part of the experiment definition.
- Fixed geometry, exact prior, phase-only far-field model; no coupling, blockage or deformation-dependent gains. This does not establish robustness to geometry errors, new shapes, or real hardware.
- Loss is mean per-scene linear NMSE for the full channel, not residual MSE. Report dB AFTER averaging linear NMSE. Also inspect p90 error.

## Outputs and interpretation

- PASTE_BACK.md: absolute scores and deformation gap.
- summary.csv: scores per seed, SNR, arm, best epoch and trained epochs.
- deformation_gap.csv: per-seed paired scene-bootstrap 95% intervals for large-minus-small error in dB. Positive means worse at large deformation. Intervals are pointwise, conditional on trained models, not confidence intervals over training seeds.
- per_scene_errors.csv: exact paired errors, channel similarity and clean channel difference.
- seed_*/arm/history.csv: training/validation learning curves and learning rates.
- bridge_diagnostics.json, geometry.pt, bridge_*.pt: reproducible physics setup.
- seed_*/arm/best.pt and last.pt: best neural weights and resumable model/optimizer/scheduler state.

Success requires good absolute large-deformation error AND a small gap to a strong small-deformation reference. A small gap caused by a poor small reference is not success. Compare the small reference with the earlier ~-22 dB result at 20 dB SNR, while remembering that these are new scenes/inputs/models. Cross-project dB values are context, not paired statistical comparisons.

If hybrid beats direct strongly, explicit geometry-based reconstruction helped. If pure physics and hybrid are similar, inspect whether refinement improves validation. If both remain far from their small counterparts, examine physics rank/prior approximation, learned model convergence and per-scene error structure. Do not conclude information loss from a neural failure alone.

## Local run and resume

```sh
python -m pip install -r requirements.txt
python verify.py
python train.py --quick --output smoke_results
python train.py --output results
python train.py --output results --resume
```

Resume requires identical settings; the last checkpoint is written atomically after each completed epoch. Scene generation is deterministic by step. An interrupted epoch restarts from its beginning. No dropout is used. Hardware/version differences may still affect numerical reproducibility. Only load trusted checkpoints; optimizer resume uses PyTorch's full checkpoint loader.

For Kaggle in the same session, set RESUME_OUTPUT to the previous printed results path and retain all settings. Across sessions, preserve the ENTIRE results directory (including .pt files), restore it under /kaggle/working, then set RESUME_OUTPUT. The compact review ZIP alone is insufficient to resume. Different settings require a fresh results folder.

## Use a trained model

`predict.py` accepts an NPZ with `observations` (complex B,8,25,25,2 raw inverse-pilot channel estimates) and `noise_variance` (positive B,8 calibrated raw complex noise variances). Use the same geometry, view order and tone pair as training. The command enforces amplitude but cannot infer whether your external geometry/order is correct. It returns a complex B,25,25,2 undeformed estimate in the original scale.

```sh
python predict.py --results results --input observations.npz --alpha 0.5 --output predictions.npz
```

## Verification checks

`verify.py` checks steering against the original NumPy physics, complex packing, paired undeformed targets, noise calibration, the reduced-rank bridge against an independent dual solve, initialization and finite training gradients. Run it before expensive experiments. QUICK additionally exercises all training, checkpoint, evaluation and reporting stages with tiny settings.
