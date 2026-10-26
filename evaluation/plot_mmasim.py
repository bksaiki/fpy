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
    # Older results also measured a "stress" workload; plot normal inputs only.
    rows = [row for row in rows if row.get("workload", "normal") == "normal"]
    for row in rows:
        row["dot_length"] = int(row["dot_length"])
        row["batch_size"] = int(row["batch_size"])
        row["threads"] = int(row["threads"])
        row["relative_iqr"] = float(row["relative_iqr"])
        row["dots_per_second"] = float(row["dots_per_second"])
    return rows


COLORS = {"fpy-core": "#2a78d6", "mmasim": "#eb6834"}
PLOTS = {
    "throughput": (
        "Complete dot products per second",
        "FPy versus eager MMA-Sim throughput",
    ),
    "speedup": ("FPy speedup over eager MMA-Sim", "FPy relative performance"),
}


def plot_points(
    rows: list[dict[str, Any]], kind: str
) -> dict[str, list[tuple[int, float]]]:
    """Select throughput series or matched FPy/MMA-Sim ratios for one design."""
    rates = {
        implementation: {
            row["batch_size"]: row["dots_per_second"]
            for row in rows
            if row["implementation"] == implementation
        }
        for implementation in COLORS
    }
    if kind == "speedup":
        fpy, mmasim = rates["fpy-core"], rates["mmasim"]
        rates = {
            "fpy-core": {
                batch: fpy[batch] / mmasim[batch]
                for batch in fpy.keys() & mmasim.keys()
            }
        }
    return {
        implementation: sorted(values.items())
        for implementation, values in rates.items()
    }


def plot_results(rows: list[dict[str, Any]], designs: list[str], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import EngFormatter, FuncFormatter

    formatters = {
        "throughput": EngFormatter(),
        "speedup": FuncFormatter(lambda value, _: f"{value:g}×"),
    }
    positions = range(len(designs))
    for length in sorted({row["dot_length"] for row in rows}):
        selected = [row for row in rows if row["dot_length"] == length]
        batches = sorted({row["batch_size"] for row in selected})
        columns = min(3, len(batches))
        height = math.ceil(len(batches) / columns)
        for kind, (ylabel, title) in PLOTS.items():
            # values[design][implementation][batch]
            values = {
                design: {
                    implementation: dict(points)
                    for implementation, points in plot_points(
                        [row for row in selected if row["design"] == design], kind
                    ).items()
                }
                for design in designs
            }
            series = ["fpy-core"] if kind == "speedup" else list(COLORS)
            width = 0.8 / len(series)
            fig, axes = plt.subplots(
                height, columns, figsize=(6 * columns, 4.2 * height), squeeze=False
            )
            for axis, batch in zip(axes.flat, batches):
                for index, implementation in enumerate(series):
                    offset = (index - (len(series) - 1) / 2) * width
                    axis.bar(
                        [position + offset for position in positions],
                        [
                            values[design][implementation].get(batch, math.nan)
                            for design in designs
                        ],
                        width * 0.9,
                        label=implementation,
                        color=COLORS[implementation],
                    )
                if kind == "speedup":
                    axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
                axis.set_title(f"batch {batch}", fontsize=10)
                axis.set_xticks(
                    list(positions), designs, rotation=60, ha="right", fontsize=7
                )
                axis.yaxis.set_major_formatter(formatters[kind])
                axis.set_ylim(bottom=0)
                axis.grid(True, axis="y", alpha=0.25)
                axis.set_axisbelow(True)
            for axis in axes.flat[len(batches) :]:
                axis.set_visible(False)
            if len(series) > 1:
                axes.flat[0].legend()
            for axis in axes[:, 0]:
                axis.set_ylabel(ylabel)
            fig.suptitle(f"{title} — length {length}")
            fig.tight_layout()
            fig.savefig(output / f"{kind}-{length}.png", dpi=180)
            plt.close(fig)


def print_headline(
    rows: list[dict[str, Any]],
    designs: list[str],
    batch: int,
    length: int,
) -> None:
    selected = {
        (row["design"], row["implementation"]): row
        for row in rows
        if row["batch_size"] == batch and row["dot_length"] == length
    }
    print(f"Headline: batch={batch}, length={length}")
    print(f"{'design':25} {'FPy dots/s':>13} {'MMA-Sim dots/s':>15} {'speedup':>9}")
    for design in designs:
        fpy = selected.get((design, "fpy-core"))
        mmasim = selected.get((design, "mmasim"))
        if not fpy or not mmasim:
            print(f"{design:25} invalid or untimed")
            continue
        speedup = fpy["dots_per_second"] / mmasim["dots_per_second"]
        unstable = max(fpy["relative_iqr"], mmasim["relative_iqr"]) > 0.10
        print(
            f"{design:25} {fpy['dots_per_second']:13.3g} "
            f"{mmasim['dots_per_second']:15.3g} {speedup:8.2f}x"
            f"{' !' if unstable else ''}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory containing summary.csv")
    parser.add_argument(
        "--output", type=Path, help="plot directory; defaults to results"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dot-length", type=int)
    args = parser.parse_args()

    results = args.results.expanduser().resolve()
    output = (args.output or results).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_summaries(results / "summary.csv")
    if not rows:
        raise SystemExit("summary.csv contains no measurements")
    designs = list(dict.fromkeys(row["design"] for row in rows))
    batch = args.batch_size or max(row["batch_size"] for row in rows)
    plot_results(rows, designs, output)
    lengths = (
        [args.dot_length]
        if args.dot_length
        else sorted({row["dot_length"] for row in rows})
    )
    for length in lengths:
        print_headline(rows, designs, batch, length)


if __name__ == "__main__":
    main()
