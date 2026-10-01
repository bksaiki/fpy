"""Plot and summarize an MMA-Sim benchmark result directory."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

EVALUATION_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(EVALUATION_ROOT / ".matplotlib"))


def read_summaries(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as source:
        rows: list[dict[str, Any]] = list(csv.DictReader(source))
    for row in rows:
        row["batch_size"] = int(row["batch_size"])
        row["threads"] = int(row["threads"])
        row["relative_iqr"] = float(row["relative_iqr"])
        row["dpa_per_second"] = float(row["dpa_per_second"])
    return rows


def plot_results(rows: list[dict[str, Any]], designs: list[str], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    implementations = ("fpy-core", "fpy-storage", "mmasim")
    colors = {"fpy-core": "#0072B2", "fpy-storage": "#009E73", "mmasim": "#D55E00"}
    for workload in sorted({row["workload"] for row in rows}):
        workload_rows = [row for row in rows if row["workload"] == workload]
        height = math.ceil(len(designs) / 4)

        fig, axes = plt.subplots(height, 4, figsize=(15, 3.2 * height), squeeze=False)
        for axis, design in zip(axes.flat, designs):
            design_rows = [row for row in workload_rows if row["design"] == design]
            for implementation in implementations:
                points = sorted(
                    (
                        row
                        for row in design_rows
                        if row["implementation"] == implementation
                    ),
                    key=lambda row: row["batch_size"],
                )
                if points:
                    axis.plot(
                        [row["batch_size"] for row in points],
                        [row["dpa_per_second"] for row in points],
                        marker="o",
                        label=implementation,
                        color=colors[implementation],
                    )
            axis.set_title(design, fontsize=9)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.25)
        for axis in axes.flat[len(designs) :]:
            axis.set_visible(False)
        axes.flat[0].legend()
        fig.supxlabel("batch size (DPAs per call)")
        fig.supylabel("DPAs per second")
        fig.suptitle(f"FPy versus eager MMA-Sim throughput — {workload}")
        fig.tight_layout()
        fig.savefig(output / f"throughput-{workload}.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(height, 4, figsize=(15, 3.2 * height), squeeze=False)
        for axis, design in zip(axes.flat, designs):
            design_rows = [row for row in workload_rows if row["design"] == design]
            by_key = {
                (row["implementation"], row["batch_size"]): row for row in design_rows
            }
            batches = sorted({row["batch_size"] for row in design_rows})
            for implementation in ("fpy-core", "fpy-storage"):
                points = [
                    (
                        batch,
                        by_key[implementation, batch]["dpa_per_second"]
                        / by_key["mmasim", batch]["dpa_per_second"],
                    )
                    for batch in batches
                    if (implementation, batch) in by_key and ("mmasim", batch) in by_key
                ]
                if points:
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        label=implementation,
                        color=colors[implementation],
                    )
            axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            axis.set_title(design, fontsize=9)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.25)
        for axis in axes.flat[len(designs) :]:
            axis.set_visible(False)
        axes.flat[0].legend()
        fig.supxlabel("batch size (DPAs per call)")
        fig.supylabel("FPy speedup over eager MMA-Sim")
        fig.suptitle(f"FPy relative performance — {workload}")
        fig.tight_layout()
        fig.savefig(output / f"speedup-{workload}.png", dpi=180)
        plt.close(fig)


def print_headline(
    rows: list[dict[str, Any]], designs: list[str], workload: str, batch: int
) -> None:
    selected = {
        (row["design"], row["implementation"]): row
        for row in rows
        if row["workload"] == workload and row["batch_size"] == batch
    }
    threads = next(row["threads"] for row in rows if row["workload"] == workload)
    print(f"Headline: workload={workload}, batch={batch}, threads={threads}")
    print(
        f"{'design':25} {'core DPA/s':>13} {'storage DPA/s':>14} {'MMA-Sim DPA/s':>15} {'core':>8} {'storage':>9}"
    )
    for design in designs:
        fpy = selected.get((design, "fpy-core"))
        storage = selected.get((design, "fpy-storage"))
        mmasim = selected.get((design, "mmasim"))
        if not fpy or not storage or not mmasim:
            print(f"{design:25} {'invalid':>64}")
            continue
        core_speedup = fpy["dpa_per_second"] / mmasim["dpa_per_second"]
        storage_speedup = storage["dpa_per_second"] / mmasim["dpa_per_second"]
        unstable = (
            max(fpy["relative_iqr"], storage["relative_iqr"], mmasim["relative_iqr"])
            > 0.10
        )
        print(
            f"{design:25} {fpy['dpa_per_second']:13.3g} {storage['dpa_per_second']:14.3g} "
            f"{mmasim['dpa_per_second']:15.3g} {core_speedup:7.2f}x "
            f"{storage_speedup:8.2f}x{' !' if unstable else ''}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory containing summary.csv")
    parser.add_argument(
        "--output", type=Path, help="plot directory; defaults to results"
    )
    parser.add_argument("--workload", default="normal")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    results = args.results.expanduser().resolve()
    output = (args.output or results).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_summaries(results / "summary.csv")
    if not rows:
        raise SystemExit("summary.csv contains no measurements")
    designs = list(dict.fromkeys(row["design"] for row in rows))
    batch = args.batch_size or max(
        row["batch_size"] for row in rows if row["workload"] == args.workload
    )
    plot_results(rows, designs, output)
    print_headline(rows, designs, args.workload, batch)


if __name__ == "__main__":
    main()
