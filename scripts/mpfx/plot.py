import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch
from pathlib import Path

from .options import CompileConfig, OptOptions


_OPTION_LABELS: dict[OptOptions, str] = {
    OptOptions(elim_round=False, allow_exact=False): 'none',
    OptOptions(elim_round=True, allow_exact=False): 'elim',
    OptOptions(elim_round=False, allow_exact=True): 'exact',
    OptOptions(elim_round=True, allow_exact=True): 'elim + exact'
}


def plot_times(
    times_by_config: dict[tuple[str, OptOptions], list[float]],
    options: list[OptOptions],
    output_dir: Path
) -> None:
    """Create a box-and-whisker plot of normalized execution times showing variance."""
    # Organize data by benchmark and option
    data_by_bench: dict[str, dict[OptOptions, list[float]]] = {}
    for config, times in times_by_config.items():
        name, opt_options = config
        if name not in data_by_bench:
            data_by_bench[name] = {}
        data_by_bench[name][opt_options] = times
    
    # Get baseline times and normalize
    normalized_data: dict[str, dict[OptOptions, list[float]]] = {}
    for bench_name, timings in data_by_bench.items():
        baseline_times = timings[options[0]]
        baseline_mean = np.mean(baseline_times)
        normalized_data[bench_name] = {
            option: [t / baseline_mean for t in times]
            for option, times in timings.items()
        }

    # Print summary statistics
    print("Normalized Execution Time Summary:")
    for bench_name, timings in normalized_data.items():
        print(f"Benchmark: {bench_name}")
        for option, times in timings.items():
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"  Option: {_OPTION_LABELS[option]:<20} Mean: {mean_time:.3f}  Std Dev: {std_time:.3f}")
    
    # Prepare data for plotting
    benchmarks = sorted(normalized_data.keys())
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Prepare box plot data
    all_data = []
    positions = []
    tick_labels = []
    colors = []
    
    # Define colors dynamically based on number of options
    color_list = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightgray', 'lightpink']
    color_map = {i: color_list[i % len(color_list)] for i in range(len(options))}
    
    pos = 1
    group_spacing = 0.4
    for bench_name in benchmarks:
        for option_idx, option in enumerate(options):
            if option in normalized_data[bench_name]:
                all_data.append(normalized_data[bench_name][option])
                positions.append(pos)
                colors.append(color_map[option_idx])
                pos += 1
        pos += group_spacing  # Add spacing between benchmark groups
    
    # Create box plot
    bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True,
                    meanprops=dict(color='black', linewidth=1.5),
                    medianprops=dict(visible=False))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Customize plot
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Normalized Runtime Time', fontsize=12)
    ax.set_title('Normalized Runtime Time per Benchmark', fontsize=14)
    
    # Set x-tick positions and labels
    tick_positions = []
    tick_labels = []
    pos = 1
    group_spacing = 0.4
    for bench_name in benchmarks:
        num_options = sum(1 for opt in options if opt in normalized_data[bench_name])
        tick_positions.append(pos + (num_options - 1) / 2)
        tick_labels.append(bench_name)
        pos += num_options + group_spacing
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Add legend
    legend_elements = [Patch(facecolor=color_map[i], label=_OPTION_LABELS[options[i]])
                      for i in range(len(options))]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add baseline reference line
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Set y-axis limits
    max_val = max(max(data) for data in all_data)
    ax.set_ylim(0, max_val * 1.1)  # Add 10% padding above max
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'normalized_execution_times.png'
    plt.savefig(plot_path, dpi=150)
    print(f'Plot saved to `{plot_path}`.')
    plt.close()


def plot_speedup(
    times_by_config: dict[tuple[str, OptOptions], list[float]],
    options: list[OptOptions],
    output_dir: Path
) -> None:
    """Create a box-and-whisker plot of speedup over baseline."""
    # Organize data by benchmark and option
    data_by_bench: dict[str, dict[OptOptions, list[float]]] = {}
    for config, times in times_by_config.items():
        name, opt_options = config
        if name not in data_by_bench:
            data_by_bench[name] = {}
        data_by_bench[name][opt_options] = times
    
    # Get baseline times and calculate speedup
    speedup_data: dict[str, dict[OptOptions, list[float]]] = {}
    for bench_name, timings in data_by_bench.items():
        baseline_times = timings[options[0]]
        baseline_mean = np.mean(baseline_times)
        speedup_data[bench_name] = {
            option: [baseline_mean / t for t in times]
            for option, times in timings.items()
        }

    # Print summary statistics
    print("\nSpeedup over Baseline Summary:")
    for bench_name, timings in speedup_data.items():
        print(f"Benchmark: {bench_name}")
        for option, times in timings.items():
            mean_speedup = np.mean(times)
            std_speedup = np.std(times)
            print(f"  Option: {_OPTION_LABELS[option]:<20} Mean: {mean_speedup:.3f}x  Std Dev: {std_speedup:.3f}")
    
    # Prepare data for plotting
    benchmarks = sorted(speedup_data.keys())
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Prepare box plot data
    all_data = []
    positions = []
    colors = []
    
    # Define colors dynamically based on number of options
    color_list = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightgray', 'lightpink']
    color_map = {i: color_list[i % len(color_list)] for i in range(len(options))}
    
    pos = 1
    group_spacing = 0.4
    for bench_name in benchmarks:
        for option_idx, option in enumerate(options):
            if option in speedup_data[bench_name]:
                all_data.append(speedup_data[bench_name][option])
                positions.append(pos)
                colors.append(color_map[option_idx])
                pos += 1
        pos += group_spacing  # Add spacing between benchmark groups
    
    # Create box plot
    bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True,
                    meanprops=dict(color='black', linewidth=1.5),
                    medianprops=dict(visible=False))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Customize plot
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Speedup over Baseline', fontsize=12)
    ax.set_title('Speedup over Baseline per Benchmark', fontsize=14)
    
    # Set x-tick positions and labels
    tick_positions = []
    tick_labels = []
    pos = 1
    group_spacing = 0.4
    for bench_name in benchmarks:
        num_options = sum(1 for opt in options if opt in speedup_data[bench_name])
        tick_positions.append(pos + (num_options - 1) / 2)
        tick_labels.append(bench_name)
        pos += num_options + group_spacing
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Add legend
    legend_elements = [Patch(facecolor=color_map[i], label=_OPTION_LABELS[options[i]])
                      for i in range(len(options))]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add baseline reference line (speedup = 1.0 means no improvement)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Set y-axis limits
    max_val = max(max(data) for data in all_data)
    ax.set_ylim(0, max_val * 1.1)  # Add 10% padding above max
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'speedup.png'
    plt.savefig(plot_path, dpi=150)
    print(f'Speedup plot saved to `{plot_path}`.')
    plt.close()
