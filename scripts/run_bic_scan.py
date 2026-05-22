from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hmm_spikes import load_mat_dataset, load_npz_dataset, run_sticky_poisson_bic_scan_isolated


def _load_dataset(path: Path, bin_size: float):
    if path.suffix.lower() == ".npz":
        return load_npz_dataset(path)
    return load_mat_dataset(path, bin_size=bin_size)


def _parse_states(text: str) -> list[int]:
    if ".." in text:
        start, stop = text.split("..", 1)
        return list(range(int(start), int(stop) + 1))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Crash-isolated serial sticky Poisson HMM BIC scan.")
    parser.add_argument("--dataset", required=True, help="Input .mat or converted .npz dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory for scan records and successful models.")
    parser.add_argument("--states", default="2..6", help="State candidates, for example '2..6' or '3,4,5'.")
    parser.add_argument("--restarts", type=int, default=10, help="Number of random restarts per K.")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--bin-size", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--timeout-s", type=float, default=None, help="Optional timeout per restart.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = _load_dataset(Path(args.dataset), args.bin_size)
    states = _parse_states(args.states)
    result = run_sticky_poisson_bic_scan_isolated(
        dataset.counts,
        states,
        bin_size=dataset.bin_size,
        n_restarts=args.restarts,
        threshold=args.threshold,
        max_iter=args.max_iter,
        timeout_s=args.timeout_s,
        base_seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"records={len(result.records)}")
    for record in result.records:
        print(
            f"K={record.n_states} restart={record.restart} status={record.status} "
            f"converged={record.converged} threshold={record.threshold_satisfied} "
            f"BIC={record.bic:.3f} LL={record.log_likelihood:.3f}"
        )
    if result.best_strict is not None:
        print(
            f"best_strict: K={result.best_strict.n_states} BIC={result.best_strict.bic:.3f} "
            f"model={result.best_strict.model_path}"
        )
    else:
        print("best_strict: none")
    if result.best_diagnostic is not None:
        print(
            f"best_diagnostic: K={result.best_diagnostic.n_states} BIC={result.best_diagnostic.bic:.3f} "
            f"status={result.best_diagnostic.status} model={result.best_diagnostic.model_path}"
        )
    else:
        print("best_diagnostic: none")


if __name__ == "__main__":
    main()
