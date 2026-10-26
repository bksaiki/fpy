"""Plot data stays independent of model imports and benchmark execution."""

from evaluation.plot_mmasim import plot_points


def test_plot_points_sort_batches_and_only_compare_matching_measurements():
    rows = [
        {"implementation": "fpy-core", "batch_size": 16, "dots_per_second": 80.0},
        {"implementation": "mmasim", "batch_size": 1, "dots_per_second": 2.0},
        {"implementation": "fpy-core", "batch_size": 1, "dots_per_second": 10.0},
        {"implementation": "mmasim", "batch_size": 256, "dots_per_second": 100.0},
    ]
    assert plot_points(rows, "throughput") == {
        "fpy-core": [(1, 10.0), (16, 80.0)],
        "mmasim": [(1, 2.0), (256, 100.0)],
    }
    assert plot_points(rows, "speedup") == {"fpy-core": [(1, 5.0)]}
    assert plot_points([], "speedup") == {"fpy-core": []}
