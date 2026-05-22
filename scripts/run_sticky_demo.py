from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hmm_spikes import load_mat_dataset, load_npz_dataset, model_log_likelihood
from hmm_spikes.phmm import fit_sticky_poisson_hmm
from hmm_spikes.plotting import plot_all_trials_summary, plot_training_diagnostics, plot_trial_decoding


def _load_dataset(path: Path, bin_size: float):
    if path.suffix.lower() == ".npz":
        return load_npz_dataset(path)
    return load_mat_dataset(path, bin_size=bin_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sticky Poisson HMM and generate paper-style figures.")
    parser.add_argument("--dataset", default="data/exampledata.mat", help="Input .mat or converted .npz dataset.")
    parser.add_argument("--output-dir", default="figures/python_sticky", help="Output directory for figures and model.")
    parser.add_argument("--states", type=int, default=3, help="Number of hidden states.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Minimum self-transition probability.")
    parser.add_argument("--bin-size", type=float, default=0.05, help="Bin size in seconds for raw MATLAB firings.")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Fraction of trials used for training.")
    parser.add_argument("--seed", type=int, default=3456, help="Random seed.")
    parser.add_argument("--max-iter", type=int, default=250, help="Maximum EM iterations.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Parameter convergence tolerance.")
    parser.add_argument("--trial-figures", type=int, default=3, help="Number of single-trial figures to save.")
    parser.add_argument("--verbose", action="store_true", help="Print training progress.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(dataset_path, args.bin_size)
    rng = np.random.default_rng(args.seed)
    n_train = int(np.ceil(dataset.n_trials * args.train_frac))
    train_ind = np.sort(rng.choice(dataset.n_trials, size=n_train, replace=False))
    val_ind = np.setdiff1d(np.arange(dataset.n_trials), train_ind)
    train_counts = [dataset.counts[i] for i in train_ind]

    print(
        f"dataset={dataset.name} trials={dataset.n_trials} neurons={dataset.n_neurons} "
        f"bins={dataset.n_bins} dt={dataset.bin_size:g}s",
        flush=True,
    )
    print(
        f"training sticky Poisson HMM with m={args.states}, theta={args.threshold:g}, train_trials={n_train}",
        flush=True,
    )

    result = fit_sticky_poisson_hmm(
        train_counts,
        args.states,
        bin_size=dataset.bin_size,
        threshold=args.threshold,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=rng,
        sticky=True,
        verbose=args.verbose,
    )

    train_ll = model_log_likelihood(train_counts, result.means, result.gamma, result.deltas)
    val_counts = [dataset.counts[i] for i in val_ind]
    val_ll = model_log_likelihood(val_counts, result.means, result.gamma) if len(val_counts) else np.nan

    np.savez_compressed(
        output_dir / "sticky_phmm_model.npz",
        means=result.means,
        rates_hz=result.rates_hz,
        gamma=result.gamma,
        deltas=result.deltas,
        log_likelihood=result.history.log_likelihood,
        gamma_diag=result.history.gamma_diag,
        reset_hits=result.history.reset_hits,
        train_ind=train_ind,
        val_ind=val_ind,
        bin_size=np.array(dataset.bin_size),
        threshold=np.array(args.threshold),
        converged=np.array(result.converged),
        threshold_satisfied=np.array(result.threshold_satisfied),
        restored_on_exit=np.array(result.restored_on_exit),
        n_iter=np.array(result.n_iter),
        train_ll=np.array(train_ll),
        val_ll=np.array(val_ll),
    )

    plot_training_diagnostics(result, path=output_dir / "training_diagnostics.png")
    plt.close("all")

    n_trial_figures = min(args.trial_figures, dataset.n_trials)
    for local_trial in range(n_trial_figures):
        firings = None if dataset.firings is None else dataset.firings[local_trial]
        plot_trial_decoding(
            firings,
            dataset.counts[local_trial],
            result,
            dataset.time_edges,
            trial_number=local_trial + 1,
            path=output_dir / f"trial_{local_trial + 1:03d}_decoding.png",
        )
        plt.close("all")

    ordered = list(train_ind) + list(val_ind)
    ordered_counts = [dataset.counts[i] for i in ordered]
    plot_all_trials_summary(
        ordered_counts,
        result,
        dataset.time_edges,
        train_cut=len(train_ind),
        path=output_dir / "all_trials_summary.png",
    )
    plt.close("all")

    print(
        f"converged={result.converged} threshold_satisfied={result.threshold_satisfied} "
        f"restored_on_exit={result.restored_on_exit} iterations={result.n_iter} "
        f"train_ll={train_ll:.3f} val_ll={val_ll:.3f}",
        flush=True,
    )
    print(f"min_diag={np.diag(result.gamma).min():.3f} max_rate={result.rates_hz.max():.2f} Hz", flush=True)
    if not result.converged:
        print(
            "warning: EM did not converge. Inspect diagnostics and rerun with more iterations, fewer states, or more restarts.",
            flush=True,
        )
    print(f"saved model and figures to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
