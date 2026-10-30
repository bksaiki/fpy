"""Plot and summarize an MMA-Sim benchmark result directory."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

EVALUATION_ROOT = Path(__file__).resolve().parents[1]  # this file is in evaluation/performance
os.environ.setdefault("MPLCONFIGDIR", str(EVALUATION_ROOT / ".matplotlib"))


def read_summaries(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as source:
        rows: list[dict[str, Any]] = list(csv.DictReader(source))
    # Older results also measured a "stress" workload; plot normal inputs only.
    rows = [row for row in rows if row.get("workload", "normal") == "normal"]
    for row in rows:
        # Older results named the torch.compile series "mmasim-compile-naive".
        if row["implementation"] == "mmasim-compile-naive":
            row["implementation"] = "mmasim-compiled"
        row["dot_length"] = int(row["dot_length"])
        row["batch_size"] = int(row["batch_size"])
        row["threads"] = int(row["threads"])
        row["relative_iqr"] = float(row["relative_iqr"])
        row["dots_per_second"] = float(row["dots_per_second"])
    return rows


COLORS = {
    "fpy-core": "#2a78d6",
    "mmasim": "#eb6834",
    "mmasim-compiled": "#d4a12a",
}
BASELINES = ("mmasim", "mmasim-compiled")
LABELS = {
    "fpy-core": "FPy",
    "mmasim": "eager MMA-Sim",
    "mmasim-compiled": "MMA-Sim, torch.compile",
}
# Short headline-table column names.
COLUMNS = {"mmasim": "eager", "mmasim-compiled": "compiled"}
PLOTS = {
    "throughput": (
        "Dot products per second",
        "FPy versus MMA-Sim throughput",
    ),
    "speedup": ("FPy speedup over MMA-Sim", "FPy relative performance"),
}


# Architecture groups for the grouped speedup plot, in display order.
ARCHITECTURES = {
    "volta": "Volta",
    "turing": "Turing",
    "ampere": "Ampere",
    "ada": "Ada",
    "hopper": "Hopper",
    "blackwell": "Blackwell",
    "rtx_blackwell": "RTX Blackwell",
    "cdna2": "CDNA2",
    "cdna3": "CDNA3",
    "fma": "FMA",
}


def architecture(design: str) -> tuple[str, str]:
    """A design's architecture group and its name within that group."""
    tokens = design.split(".")
    if tokens[-1] == "fma":
        return "fma", tokens[0]
    return tokens[1], ".".join(tokens[2:])


# MMA-Sim's arithmetic models, in legend order, with their bar colors.
ARITHMETIC = {
    "T-FDPA": "#2a78d6",
    "ST-FDPA": "#eb6834",
    "GST-FDPA": "#1baf7a",
    "TR-FDPA": "#eda100",
    "GTR-FDPA": "#e87ba4",
    "FTZ mul-add": "#008300",
    "FMA": "#6250d6",
}


def arithmetic(design: str) -> str:
    """The MMA-Sim arithmetic model that a design transcribes."""
    group, name = architecture(design)
    if group == "fma":
        return "FMA"
    if group == "cdna2":
        return "FTZ mul-add"
    if group == "cdna3":
        return "GTR-FDPA" if "8" in name else "TR-FDPA"
    if name in ("nvfp4", "mxfp4"):
        return "GST-FDPA"
    if name.startswith("mx."):
        return "ST-FDPA"
    return "T-FDPA"


# Input formats, widest first, with their bar colors. Designs whose operands
# mix FP8 and FP4 count as FP4.
INPUT_FORMATS = {
    "FP64": "#2a78d6",
    "FP32": "#eb6834",
    "TF32": "#1baf7a",
    "BF16": "#eda100",
    "FP16": "#e87ba4",
    "FP8": "#008300",
    "FP4": "#6250d6",
    "MX / NVFP4": "#e34948",
}


def input_format(design: str) -> str:
    """The input format of a design's operands."""
    group, name = architecture(design)
    if group == "fma":
        return name.upper()
    tokens = name.split(".")
    if tokens[0] in ("mx", "mxfp4", "nvfp4"):
        return "MX / NVFP4"
    if "e2m1" in tokens:
        return "FP4"
    if {"e4m3", "e5m2", "fp8", "bf8"} & set(tokens):
        return "FP8"
    return {"tf32": "TF32", "bf16": "BF16", "bf16_1k": "BF16", "f16": "FP16"}[
        tokens[0]
    ]


def plot_points(
    rows: list[dict[str, Any]], kind: str
) -> dict[str, list[tuple[int, float]]]:
    """Select throughput series or matched FPy/MMA-Sim ratios for one design.

    Speedup series are keyed by the MMA-Sim baseline they divide by.
    """
    rates = {
        implementation: {
            row["batch_size"]: row["dots_per_second"]
            for row in rows
            if row["implementation"] == implementation
        }
        for implementation in COLORS
    }
    if kind == "speedup":
        fpy = rates["fpy-core"]
        rates = {
            baseline: {
                batch: fpy[batch] / rates[baseline][batch]
                for batch in fpy.keys() & rates[baseline].keys()
            }
            for baseline in BASELINES
        }
    return {
        implementation: sorted(values.items())
        for implementation, values in rates.items()
        if values
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
        # Many designs get one full-width panel per batch size.
        columns = 1 if len(designs) > 20 else min(3, len(batches))
        panel_width = max(6.0, 0.3 * len(designs))
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
            measured = {
                implementation
                for design in values.values()
                for implementation in design
            }
            series = [
                implementation
                for implementation in (BASELINES if kind == "speedup" else COLORS)
                if implementation in measured
            ]
            width = (0.85 if kind == "speedup" else 0.8) / len(series)
            # Speedup bars sit nearly touching in a compact panel.
            kind_width = (
                max(6.0, 0.15 * len(designs)) if kind == "speedup" else panel_width
            )
            fig, axes = plt.subplots(
                height, columns, figsize=(kind_width * columns, 5.5 * height), squeeze=False
            )
            for axis, batch in zip(axes.flat, batches):
                order = designs
                if kind == "speedup":
                    # Smallest speedup over the first baseline first; unmeasured last.
                    def speedup(design: str) -> float:
                        value = values[design].get(series[0], {}).get(batch, math.nan)
                        return math.inf if math.isnan(value) else value

                    order = sorted(designs, key=speedup)
                for index, implementation in enumerate(series):
                    offset = (index - (len(series) - 1) / 2) * width
                    heights = [
                        values[design].get(implementation, {}).get(batch, math.nan)
                        for design in order
                    ]
                    bars = axis.bar(
                        [position + offset for position in positions],
                        heights,
                        width if kind == "speedup" else width * 0.9,
                        label=(
                            f"over {LABELS[implementation]}"
                            if kind == "speedup"
                            else LABELS[implementation]
                        ),
                        color=COLORS[implementation],
                    )
                    if kind == "throughput" and implementation in BASELINES:
                        # Label each MMA-Sim bar with FPy's speedup over it,
                        # aligned at the top of the design's bar group.
                        for bar, design, h in zip(bars, order, heights):
                            f = values[design].get("fpy-core", {}).get(batch, math.nan)
                            if math.isnan(f) or math.isnan(h):
                                continue
                            top = max(
                                values[design].get(other, {}).get(batch, 0.0)
                                for other in series
                            )
                            axis.annotate(
                                f"{f / h:.0f}×" if f / h >= 10 else f"{f / h:.2g}×",
                                (bar.get_x() + bar.get_width() / 2, top),
                                xytext=(0, 2),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                rotation=90,
                                fontsize=6,
                                color=COLORS[implementation],
                            )
                if kind == "throughput":
                    axis.margins(y=0.18)
                if kind == "speedup":
                    axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
                axis.set_title(f"batch {batch}", fontsize=10)
                axis.set_xticks(
                    list(positions), order, rotation=90, ha="center",
                    fontsize=6 if kind == "speedup" else 7,
                )

                shown = [
                    h
                    for design in order
                    for implementation in series
                    if not math.isnan(
                        h := values[design].get(implementation, {}).get(batch, math.nan)
                    )
                ]
                if kind == "speedup" and shown and max(shown) > 20 * min(shown):
                    # Small batches span from about 1x to thousands of x.
                    axis.set_yscale("log")
                else:
                    axis.set_ylim(bottom=0)
                axis.yaxis.set_major_formatter(formatters[kind])
                axis.grid(True, axis="y", alpha=0.25)
                axis.set_axisbelow(True)
            for axis in axes.flat[len(batches) :]:
                axis.set_visible(False)
            for axis in axes[:, 0]:
                axis.set_ylabel(ylabel)
            fig.suptitle(f"{title} — length {length}")
            if len(series) > 1:
                # Keep the legend above the panels so it never covers a bar.
                fig.legend(
                    *axes.flat[0].get_legend_handles_labels(),
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.95),
                    ncol=len(series),
                    frameon=False,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.93))
            else:
                # Keep the figure title clear of the first panel's title.
                fig.tight_layout(rect=(0, 0, 1, 0.98))
            for suffix in ("png", "pdf"):
                fig.savefig(output / f"{kind}-{length}.{suffix}", dpi=180)
            plt.close(fig)


def speedups(
    rows: list[dict[str, Any]], designs: list[str], length: int
) -> dict[str, dict[int, float]]:
    """FPy's speedup over eager MMA-Sim, by design and batch size."""
    selected = [row for row in rows if row["dot_length"] == length]
    return {
        design: dict(
            plot_points(
                [row for row in selected if row["design"] == design], "speedup"
            ).get("mmasim", [])
        )
        for design in designs
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(map(math.log, values)) / len(values))


def plot_speedup_variants(
    rows: list[dict[str, Any]], designs: list[str], output: Path
) -> None:
    """Alternative layouts of FPy's speedup over eager MMA-Sim.

    `speedup-batches-<length>`: one panel, one bar per batch size for each
    design, side by side, designs sorted by their geometric-mean speedup.
    `speedup-stacked-<length>`: the same, with each design's bars overlaid
    from zero, shortest in front. Speedups do not add, so the bars are not
    summed.
    `speedup-grouped-<length>`: one panel per batch size, designs grouped by
    architecture in a fixed order.
    `speedup-models-<length>`: one panel per batch size, designs sorted by
    speedup and colored by their MMA-Sim arithmetic model;
    `speedup-formats-<length>`, the same colored by input format.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    times = FuncFormatter(lambda value, _: f"{value:g}×")
    ylabel = PLOTS["speedup"][0]
    for length in sorted({row["dot_length"] for row in rows}):
        values = speedups(rows, designs, length)
        measured = [design for design in designs if values[design]]
        if not measured:
            continue
        batches = sorted({batch for design in measured for batch in values[design]})
        # Light to dark: larger batches are darker.
        cmap = plt.get_cmap("Blues")
        shades = {
            batch: cmap(0.45 + 0.5 * index / max(1, len(batches) - 1))
            for index, batch in enumerate(batches)
        }

        # One panel, a bar per batch size, side by side or overlaid.
        order = sorted(
            measured, key=lambda design: geomean(list(values[design].values()))
        )
        for name, stacked in (("batches", False), ("stacked", True)):
            fig, axis = plt.subplots(
                figsize=(max(6.0, (0.15 if stacked else 0.2) * len(order)), 5.4)
            )
            if stacked:
                for position, design in enumerate(order):
                    # Tallest first, so each shorter bar is drawn in front.
                    for batch, value in sorted(
                        values[design].items(), key=lambda item: -item[1]
                    ):
                        axis.bar(position, value, 0.85, color=shades[batch])
                for batch in batches:
                    axis.bar(0, 0, color=shades[batch], label=f"batch {batch}")
            else:
                width = 0.85 / len(batches)
                for index, batch in enumerate(batches):
                    offset = (index - (len(batches) - 1) / 2) * width
                    shown = [
                        (position + offset, values[design][batch])
                        for position, design in enumerate(order)
                        if batch in values[design]
                    ]
                    axis.bar(
                        *zip(*shown),
                        width,
                        color=shades[batch],
                        label=f"batch {batch}",
                    )
            axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            axis.set_xticks(range(len(order)), order, rotation=90, fontsize=6)
            axis.set_xlim(-0.6, len(order) - 0.4)
            axis.set_ylim(bottom=0)
            axis.yaxis.set_major_formatter(times)
            axis.set_ylabel(ylabel)
            axis.grid(True, axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(fontsize=7, frameon=False, loc="upper left")
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                fig.savefig(output / f"speedup-{name}-{length}.{suffix}", dpi=180)
            plt.close(fig)

        # One panel per batch size, designs grouped by architecture.
        order = sorted(
            measured,
            key=lambda design: list(ARCHITECTURES).index(architecture(design)[0]),
        )
        groups = [architecture(design)[0] for design in order]
        bounds = [
            index
            for index in range(1, len(order))
            if groups[index] != groups[index - 1]
        ]
        fig, axes = plt.subplots(
            len(batches),
            1,
            figsize=(max(6.0, 0.15 * len(order)), 2.2 * len(batches) + 1.4),
            sharex=True,
            squeeze=False,
        )
        for axis, batch in zip(axes[:, 0], batches):
            heights = [values[design].get(batch, math.nan) for design in order]
            axis.bar(range(len(order)), heights, 0.85, color=COLORS["mmasim"])
            for bound in bounds:
                axis.axvline(bound - 0.5, color="#888888", linewidth=0.6)
            axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            axis.set_ylabel(f"batch {batch}", fontsize=9)
            axis.set_xlim(-0.6, len(order) - 0.4)
            axis.set_ylim(bottom=0)
            axis.yaxis.set_major_formatter(times)
            axis.grid(True, axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        # Architecture names above the first panel, short names below the last.
        top = axes[0, 0].secondary_xaxis("top")
        starts = [0, *bounds]
        ends = [*bounds, len(order)]
        # Alternate label heights so narrow neighboring groups never overlap.
        top.set_xticks(
            [(start + end - 1) / 2 for start, end in zip(starts, ends)],
            [
                ("\n" if index % 2 else "") + ARCHITECTURES[groups[start]]
                + ("" if index % 2 else "\n")
                for index, start in enumerate(starts)
            ],
            fontsize=7,
        )
        top.tick_params(length=0, pad=2)
        top.spines["top"].set_visible(False)
        axes[-1, 0].set_xticks(
            range(len(order)),
            [architecture(design)[1] for design in order],
            rotation=90,
            fontsize=6,
        )
        fig.supylabel(ylabel, fontsize=10)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"speedup-grouped-{length}.{suffix}", dpi=180)
        plt.close(fig)

        # One panel per batch size, sorted ascending, colored by a category.
        for name, colors, category in (
            ("models", ARITHMETIC, arithmetic),
            ("formats", INPUT_FORMATS, input_format),
        ):
            fig, axes = plt.subplots(
                len(batches),
                1,
                figsize=(max(6.0, 0.15 * len(measured)), 3.4 * len(batches) + 0.6),
                squeeze=False,
            )
            for axis, batch in zip(axes[:, 0], batches):
                order = sorted(
                    (design for design in measured if batch in values[design]),
                    key=lambda design: values[design][batch],
                )
                axis.bar(
                    range(len(order)),
                    [values[design][batch] for design in order],
                    0.8,
                    color=[colors[category(design)] for design in order],
                )
                axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
                axis.set_title(f"batch {batch}", fontsize=10)
                axis.set_xticks(range(len(order)), order, rotation=90, fontsize=6)
                axis.set_xlim(-0.6, len(order) - 0.4)
                axis.set_ylim(bottom=0)
                axis.yaxis.set_major_formatter(times)
                axis.set_ylabel(ylabel, fontsize=9)
                axis.grid(True, axis="y", alpha=0.25)
                axis.set_axisbelow(True)
                axis.spines[["top", "right"]].set_visible(False)
            # Only categories with a measured design appear in the legend.
            present = {category(design) for design in measured}
            shown = [label for label in colors if label in present]
            fig.legend(
                [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in shown],
                shown,
                loc="upper center",
                ncol=len(shown),
                frameon=False,
                fontsize=8,
            )
            fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
            for suffix in ("png", "pdf"):
                fig.savefig(output / f"speedup-{name}-{length}.{suffix}", dpi=180)
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
    measured = {implementation for _, implementation in selected}
    baselines = [baseline for baseline in BASELINES if baseline in measured]
    print(f"Headline: batch={batch}, length={length} (FPy speedup over each MMA-Sim)")
    print(
        f"{'design':25} {'FPy dots/s':>11}"
        + "".join(f" {COLUMNS[b] + ' dots/s':>16}" for b in baselines)
        + "".join(f" {'vs ' + COLUMNS[b]:>12}" for b in baselines)
    )
    for design in designs:
        fpy = selected.get((design, "fpy-core"))
        if not fpy:
            print(f"{design:25} invalid or untimed")
            continue
        # A baseline may be missing when its compilation failed validation.
        others = [selected.get((design, baseline)) for baseline in baselines]
        rates = "".join(
            f" {row['dots_per_second']:16.3g}" if row else f" {'-':>16}"
            for row in others
        )
        speedups = "".join(
            f" {fpy['dots_per_second'] / row['dots_per_second']:11.2f}x"
            if row
            else f" {'-':>12}"
            for row in others
        )
        unstable = max(row["relative_iqr"] for row in (fpy, *others) if row) > 0.10
        print(
            f"{design:25} {fpy['dots_per_second']:11.3g}{rates}{speedups}"
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
    plot_speedup_variants(rows, designs, output)
    lengths = (
        [args.dot_length]
        if args.dot_length
        else sorted({row["dot_length"] for row in rows})
    )
    for length in lengths:
        print_headline(rows, designs, batch, length)


if __name__ == "__main__":
    main()
