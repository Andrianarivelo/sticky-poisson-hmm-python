from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hmm_spikes import load_mat_dataset, save_npz_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert bundled MATLAB spike datasets to NumPy npz files.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[
            "data/exampledata.mat",
            "data/MMPP_N20m10T15_50trials_MCdt50ms_maxfr30_initstate0_3.mat",
        ],
        help="MATLAB .mat files to convert.",
    )
    parser.add_argument("--output-dir", default="data/python", help="Directory for converted npz files.")
    parser.add_argument("--bin-size", type=float, default=0.05, help="Bin size in seconds for raw firing data.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for input_path in args.inputs:
        dataset = load_mat_dataset(input_path, bin_size=args.bin_size)
        dt_ms = int(round(dataset.bin_size * 1000))
        out_path = output_dir / f"{Path(input_path).stem}_dt{dt_ms}ms.npz"
        save_npz_dataset(dataset, out_path)
        print(
            f"wrote {out_path} | trials={dataset.n_trials} "
            f"neurons={dataset.n_neurons} bins={dataset.n_bins} dt={dataset.bin_size:g}s"
        )


if __name__ == "__main__":
    main()
