from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from .phmm import fit_sticky_poisson_hmm
from .selection import load_counts_archive, poisson_hmm_bic


def main() -> int:
    parser = argparse.ArgumentParser(description="Internal isolated sticky Poisson HMM fit worker.")
    parser.add_argument("--counts", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--states", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    model_path = Path(args.model)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        counts, bin_size = load_counts_archive(args.counts)
        result = fit_sticky_poisson_hmm(
            counts,
            args.states,
            bin_size=bin_size,
            threshold=args.threshold,
            max_iter=args.max_iter,
            random_state=args.seed,
        )
        total_bins = sum(trial.shape[1] for trial in counts)
        bic = poisson_hmm_bic(result.log_likelihood, args.states, counts[0].shape[0], total_bins)
        min_diag = float(np.diag(result.gamma).min())
        if result.converged and result.threshold_satisfied:
            status = "strict"
        elif result.threshold_satisfied:
            status = "diagnostic"
        else:
            status = "invalid"

        np.savez_compressed(
            model_path,
            means=result.means,
            rates_hz=result.rates_hz,
            gamma=result.gamma,
            deltas=result.deltas,
            log_likelihood=np.asarray(result.log_likelihood),
            bic=np.asarray(bic),
            converged=np.asarray(result.converged),
            threshold_satisfied=np.asarray(result.threshold_satisfied),
            restored_on_exit=np.asarray(result.restored_on_exit),
            n_iter=np.asarray(result.n_iter),
            threshold=np.asarray(args.threshold),
            bin_size=np.asarray(bin_size),
            n_states=np.asarray(args.states),
            seed=np.asarray(args.seed),
        )
        summary = {
            "n_states": args.states,
            "seed": args.seed,
            "status": status,
            "converged": result.converged,
            "threshold_satisfied": result.threshold_satisfied,
            "restored_on_exit": result.restored_on_exit,
            "n_iter": result.n_iter,
            "log_likelihood": result.log_likelihood,
            "bic": bic,
            "min_diag": min_diag,
            "message": "",
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        summary = {
            "n_states": args.states,
            "seed": args.seed,
            "status": "error",
            "converged": False,
            "threshold_satisfied": False,
            "restored_on_exit": False,
            "n_iter": 0,
            "log_likelihood": float("-inf"),
            "bic": float("inf"),
            "min_diag": float("nan"),
            "message": f"{exc}\n{traceback.format_exc()}",
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 2


if __name__ == "__main__":
    sys.exit(main())
