"""Plot and summarize an MMA-Sim benchmark result directory.

Run from the repository root:

    ./.venv/bin/python evaluation/performance/plot_mmasim.py evaluation/results/<run-id>
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from itertools import product
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


def plot_results(
    rows: list[dict[str, Any]], designs: list[str], output: Path, tag: str = ""
) -> None:
    """`throughput` and `speedup`: one panel per configuration, with a row per
    dot length and a column per batch size. Unmeasured configurations stay blank.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import EngFormatter, FuncFormatter

    formatters = {
        "throughput": EngFormatter(),
        "speedup": FuncFormatter(lambda value, _: f"{value:g}×"),
    }
    positions = range(len(designs))
    lengths = sorted({row["dot_length"] for row in rows})
    batches = sorted({row["batch_size"] for row in rows})
    panel_width = max(6.0, 0.3 * len(designs))
    for kind, (ylabel, title) in PLOTS.items():
        # values[length][design][implementation][batch]
        values = {
            length: {
                design: {
                    implementation: dict(points)
                    for implementation, points in plot_points(
                        [
                            row
                            for row in rows
                            if row["dot_length"] == length and row["design"] == design
                        ],
                        kind,
                    ).items()
                }
                for design in designs
            }
            for length in lengths
        }
        measured = {
            implementation
            for by_design in values.values()
            for design in by_design.values()
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
            len(lengths),
            len(batches),
            figsize=(kind_width * len(batches), 5.5 * len(lengths)),
            squeeze=False,
        )
        for (row_index, length), (column, batch) in product(
            enumerate(lengths), enumerate(batches)
        ):
            axis = axes[row_index, column]
            panel = values[length]
            if not any(
                batch in points for design in panel.values() for points in design.values()
            ):
                axis.set_visible(False)
                continue
            order = designs
            if kind == "speedup":
                # Smallest speedup over the first baseline first; unmeasured last.
                def speedup(design: str) -> float:
                    value = panel[design].get(series[0], {}).get(batch, math.nan)
                    return math.inf if math.isnan(value) else value

                order = sorted(designs, key=speedup)
            for index, implementation in enumerate(series):
                offset = (index - (len(series) - 1) / 2) * width
                heights = [
                    panel[design].get(implementation, {}).get(batch, math.nan)
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
                if kind == "speedup":
                    mark_geomean(axis, heights, COLORS[implementation])
                if kind == "throughput" and implementation in BASELINES:
                    # Label each MMA-Sim bar with FPy's speedup over it,
                    # aligned at the top of the design's bar group.
                    for bar, design, h in zip(bars, order, heights):
                        f = panel[design].get("fpy-core", {}).get(batch, math.nan)
                        if math.isnan(f) or math.isnan(h):
                            continue
                        top = max(
                            panel[design].get(other, {}).get(batch, 0.0)
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
            axis.set_title(f"length {length}, batch {batch}", fontsize=10)
            axis.set_xticks(
                list(positions), order, rotation=90, ha="center",
                fontsize=6 if kind == "speedup" else 7,
            )

            shown = [
                h
                for design in order
                for implementation in series
                if not math.isnan(
                    h := panel[design].get(implementation, {}).get(batch, math.nan)
                )
            ]
            if kind == "speedup" and shown and max(shown) > 20 * min(shown):
                # Small batches span from about 1x to thousands of x.
                axis.set_yscale("log")
            else:
                axis.set_ylim(bottom=0)
            axis.yaxis.set_major_formatter(formatters[kind])
            axis.set_ylabel(ylabel)
            axis.grid(True, axis="y", alpha=0.25)
            axis.set_axisbelow(True)
        # Inches from the top, so the header stays compact in a tall grid.
        figure_height = fig.get_figheight()
        fig.suptitle(title, y=1 - 0.1 / figure_height, va="top")
        if len(series) > 1:
            # Keep the legend above the panels so it never covers a bar.
            first = next(axis for axis in axes.flat if axis.get_visible())
            fig.legend(
                *first.get_legend_handles_labels(),
                loc="upper center",
                bbox_to_anchor=(0.5, 1 - 0.35 / figure_height),
                ncol=len(series),
                frameon=False,
            )
            fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / figure_height))
        else:
            # Keep the figure title clear of the first panel's title.
            fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / figure_height))
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{kind}{tag}.{suffix}", dpi=180)
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


def mark_geomean(axis: Any, heights: list[float], color: str = "black") -> None:
    """Draw and label the geometric mean of the measured *heights* on *axis*."""
    measured = [height for height in heights if not math.isnan(height)]
    if not measured:
        return
    mean = geomean(measured)
    axis.axhline(mean, color=color, linewidth=1.0, linestyle=":")
    axis.annotate(
        f"geomean {mean:.2f}×",
        (0, mean),
        xycoords=axis.get_yaxis_transform(),
        xytext=(3, 2),
        textcoords="offset points",
        fontsize=7,
        color=color,
    )


def plot_speedup_lengths(
    rows: list[dict[str, Any]], designs: list[str], output: Path
) -> None:
    """`speedup-lengths`: FPy's speedup over eager MMA-Sim, one panel per batch
    size with one bar per dot length for each design, designs sorted by their
    geometric-mean speedup.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    lengths = sorted({row["dot_length"] for row in rows})
    batches = sorted({row["batch_size"] for row in rows})
    by_length = {length: speedups(rows, designs, length) for length in lengths}
    # Light to dark: longer vectors are darker.
    cmap = plt.get_cmap("Blues")
    shades = {
        length: cmap(0.45 + 0.5 * index / max(1, len(lengths) - 1))
        for index, length in enumerate(lengths)
    }
    fig, axes = plt.subplots(
        len(batches),
        1,
        figsize=(max(6.0, 0.2 * len(designs)), 5.4 * len(batches)),
        squeeze=False,
    )
    for axis, batch in zip(axes[:, 0], batches):
        # values[design][length]
        values = {
            design: {
                length: by_length[length][design][batch]
                for length in lengths
                if batch in by_length[length][design]
            }
            for design in designs
        }
        order = sorted(
            (design for design in designs if values[design]),
            key=lambda design: geomean(list(values[design].values())),
        )
        width = 0.85 / len(lengths)
        for index, length in enumerate(lengths):
            offset = (index - (len(lengths) - 1) / 2) * width
            shown = [
                (position + offset, values[design][length])
                for position, design in enumerate(order)
                if length in values[design]
            ]
            if not shown:
                continue
            positions, heights = zip(*shown)
            axis.bar(
                positions,
                heights,
                width,
                color=shades[length],
                label=f"length {length} (geomean {geomean(list(heights)):.2f}×)",
            )
        axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(f"batch {batch}", fontsize=10)
        axis.set_xticks(range(len(order)), order, rotation=90, fontsize=6)
        axis.set_xlim(-0.6, len(order) - 0.4)
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
        axis.set_ylabel(PLOTS["speedup"][0])
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"speedup-lengths.{suffix}", dpi=180)
    plt.close(fig)


def plot_speedup_variants(
    rows: list[dict[str, Any]], designs: list[str], output: Path, tag: str = ""
) -> None:
    """Alternative layouts of FPy's speedup over eager MMA-Sim.

    `speedup-batches`: one panel per dot length, one bar per batch size for
    each design, side by side, designs sorted by their geometric-mean speedup.
    `speedup-stacked`: the same, with each design's bars overlaid from zero,
    shortest in front. Speedups do not add, so the bars are not summed.

    The rest have one panel per configuration, with a row per dot length and
    a column per batch size. Unmeasured configurations stay blank.
    `speedup-grouped`: designs grouped by architecture in a fixed order.
    `speedup-models`: designs sorted by speedup and colored by their MMA-Sim
    arithmetic model; `speedup-formats`, the same colored by input format.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    times = FuncFormatter(lambda value, _: f"{value:g}×")
    ylabel = PLOTS["speedup"][0]
    lengths = sorted({row["dot_length"] for row in rows})
    batches = sorted({row["batch_size"] for row in rows})
    by_length = {length: speedups(rows, designs, length) for length in lengths}
    measured = [
        design
        for design in designs
        if any(by_length[length][design] for length in lengths)
    ]
    if not measured:
        return
    # Light to dark: larger batches are darker.
    cmap = plt.get_cmap("Blues")
    shades = {
        batch: cmap(0.45 + 0.5 * index / max(1, len(batches) - 1))
        for index, batch in enumerate(batches)
    }
    # The measured configurations, by the grid position of their panel.
    panels = {
        (row, column): (length, batch)
        for (row, length), (column, batch) in product(
            enumerate(lengths), enumerate(batches)
        )
        if any(batch in by_length[length][design] for design in measured)
    }

    # One panel per dot length, a bar per batch size, side by side or overlaid.
    for name, stacked in (("batches", False), ("stacked", True)):
        fig, axes = plt.subplots(
            len(lengths),
            1,
            figsize=(
                max(6.0, (0.15 if stacked else 0.2) * len(measured)),
                5.4 * len(lengths),
            ),
            squeeze=False,
        )
        for axis, length in zip(axes[:, 0], lengths):
            values = by_length[length]
            order = sorted(
                (design for design in measured if values[design]),
                key=lambda design: geomean(list(values[design].values())),
            )
            labels = {
                batch: "batch {} (geomean {:.2f}×)".format(
                    batch,
                    geomean(shown),
                )
                for batch in batches
                if (
                    shown := [
                        values[design][batch]
                        for design in order
                        if batch in values[design]
                    ]
                )
            }
            if stacked:
                for position, design in enumerate(order):
                    # Tallest first, so each shorter bar is drawn in front.
                    for batch, value in sorted(
                        values[design].items(), key=lambda item: -item[1]
                    ):
                        axis.bar(position, value, 0.85, color=shades[batch])
                for batch in labels:
                    axis.bar(0, 0, color=shades[batch], label=labels[batch])
            else:
                width = 0.85 / len(batches)
                for index, batch in enumerate(batches):
                    if batch not in labels:
                        continue
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
                        label=labels[batch],
                    )
            axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            axis.set_title(f"length {length}", fontsize=10)
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
            fig.savefig(output / f"speedup-{name}{tag}.{suffix}", dpi=180)
        plt.close(fig)

    # One panel per configuration, designs grouped by architecture.
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
    starts = [0, *bounds]
    ends = [*bounds, len(order)]
    fig, axes = plt.subplots(
        len(lengths),
        len(batches),
        figsize=(
            max(6.0, 0.15 * len(order)) * len(batches),
            2.6 * len(lengths) + 1.4,
        ),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.set_visible(False)
    for (row, column), (length, batch) in panels.items():
        axis = axes[row, column]
        axis.set_visible(True)
        values = by_length[length]
        heights = [values[design].get(batch, math.nan) for design in order]
        axis.bar(range(len(order)), heights, 0.85, color=COLORS["mmasim"])
        mark_geomean(axis, heights)
        for bound in bounds:
            axis.axvline(bound - 0.5, color="#888888", linewidth=0.6)
        axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(f"length {length}, batch {batch}", fontsize=9)
        axis.set_xlim(-0.6, len(order) - 0.4)
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_formatter(times)
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        # Architecture names above each column, short names below it.
        column_rows = [other for other, c in panels if c == column]
        if row == min(column_rows):
            top = axis.secondary_xaxis("top")
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
        if row == max(column_rows):
            axis.set_xticks(
                range(len(order)),
                [architecture(design)[1] for design in order],
                rotation=90,
                fontsize=6,
            )
        else:
            axis.set_xticks(range(len(order)), [])
    fig.supylabel(ylabel, fontsize=10)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"speedup-grouped{tag}.{suffix}", dpi=180)
    plt.close(fig)

    # One panel per configuration, sorted ascending, colored by a category.
    for name, colors, category in (
        ("models", ARITHMETIC, arithmetic),
        ("formats", INPUT_FORMATS, input_format),
    ):
        fig, axes = plt.subplots(
            len(lengths),
            len(batches),
            figsize=(
                max(6.0, 0.15 * len(measured)) * len(batches),
                3.4 * len(lengths) + 0.6,
            ),
            squeeze=False,
        )
        for axis in axes.flat:
            axis.set_visible(False)
        for (row, column), (length, batch) in panels.items():
            axis = axes[row, column]
            axis.set_visible(True)
            values = by_length[length]
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
            mark_geomean(axis, [values[design][batch] for design in order])
            axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            axis.set_title(f"length {length}, batch {batch}", fontsize=10)
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
            fig.savefig(output / f"speedup-{name}{tag}.{suffix}", dpi=180)
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
    # Over the designs measured by both FPy and the baseline.
    means = []
    for baseline in baselines:
        ratios = [
            selected[design, "fpy-core"]["dots_per_second"]
            / selected[design, baseline]["dots_per_second"]
            for design in designs
            if (design, "fpy-core") in selected and (design, baseline) in selected
        ]
        means.append(f" {geomean(ratios):11.2f}x" if ratios else f" {'-':>12}")
    print(f"{'geomean':25} {'':>11}{' ' * 17 * len(baselines)}{''.join(means)}")


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
    # One set of figures per dot length, named `<plot>-<length>`.
    for length in sorted({row["dot_length"] for row in rows}):
        selected = [row for row in rows if row["dot_length"] == length]
        plot_results(selected, designs, output, f"-{length}")
        plot_speedup_variants(selected, designs, output, f"-{length}")
    plot_speedup_lengths(rows, designs, output)
    lengths = (
        [args.dot_length]
        if args.dot_length
        else sorted({row["dot_length"] for row in rows})
    )
    for length in lengths:
        print_headline(rows, designs, batch, length)


if __name__ == "__main__":
    main()
