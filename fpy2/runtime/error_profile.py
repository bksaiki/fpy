from ..ir import Expr
from typing import Any, Literal
import numpy as np

Verbosity = Literal["minimal", "standard", "detailed"]

class ErrorProfile:
    """
    A class to store and analyze the errors in a set of sampled data, 
    including the skipped samples and their associated error statistics.
    """

    samples: list[Any]

    skipped_samples: list[Any]

    errors: dict[Expr, list[float]]

    def __init__(self, samples, skipped_samples, errors: dict[Expr, list[float]]):
        self.samples = samples
        self.skipped_samples = skipped_samples
        self.errors = errors

    def _compute_statistics(self, stats: list[float], verbosity: Verbosity) -> dict[str, float]:
        """
        Computes statistics based on the provided list of error values with given verbosity
        """
        stats = np.array(stats)
        mean = np.mean(stats)
        statistics = {"Mean": mean}
        
        if verbosity in ["standard", "detailed"]:
            statistics.update({
                "Median": np.median(stats),
                "Min": np.min(stats),
                "Max": np.max(stats)
            })
        
        if verbosity == "detailed":
            statistics.update({
                "Q1": np.percentile(stats, 25),
                "Q3": np.percentile(stats, 75),
                "Std Dev": np.std(stats, ddof=1)
            })
        
        return statistics

    def print_summary(self, verbosity: Verbosity = "standard", decimal_places = 4) -> None:
        """
        Prints a summary of the error profile including statistics for each expression.
        """

        num_samples       = len(self.samples)
        num_skipped       = len(self.skipped_samples)
        percent_samples   = ((num_samples - num_skipped) / num_samples) * 100 if num_samples > 0 else 0
        total_expressions = len(self.errors)

        # Compute mean errors and categorize expressions
        error_expr = []
        no_error_exprs = []
        
        for expr, eval in self.errors.items():
            mean_error = np.mean(eval).item()
            if mean_error > 0:
                error_expr.append((expr, eval, mean_error))
            else:
                no_error_exprs.append((expr, len(eval)))

        # Sort expressions by descending mean error
        error_expr.sort(key=lambda x: x[2], reverse=True)

        # Compute percentage of expressions with non-zero errors
        num_error_expr = len(error_expr)
        percent_error_expr = (num_error_expr / total_expressions) * 100 if total_expressions > 0 else 0
        
        print("=" * 40)
        print(" Expression Profiler Summary".center(40))
        print("=" * 40)
        print(f"Evaluated Sampled points: {num_samples - num_skipped} / {num_samples} ({percent_samples:.2f}%)")
        print(f"Expressions with errors : {num_error_expr} / {total_expressions} ({percent_error_expr:.2f}%)\n")
        
        if error_expr:
            print(f"Expressions with errors (sorted by mean error in descending order):")
            print("=" * 40)

        for idx, (expr, eval, _) in enumerate(error_expr, start=1):
            print(f"{idx}. {expr.format()}")
            print(f"  Number of evaluations: {len(eval)}")
            
            if eval:
                statistics = self._compute_statistics(eval, verbosity)
                
                print("  Error Stats:")

                # Enforce ordering of error keys
                ordered_keys = ["Min", "Q1", "Median", "Mean", "Q3", "Max", "Std Dev"]
                if verbosity == "standard":
                    ordered_keys = ["Min", "Median", "Mean", "Max"]
                elif verbosity == "minimal":
                    ordered_keys = ["Mean"]
                
                max_key_length = max(len(key) for key in ordered_keys if key in statistics)
                for key in ordered_keys:
                    if key in statistics:
                        print(f"    {key.ljust(max_key_length)} : {statistics[key]:.{decimal_places}f}") 
            else:
                print("  No evaluations available.")
            
            print()

        # Print expressions with zero errors at the end
        if no_error_exprs:
            print(f"Expressions with no errors:")
            print("=" * 40)
            for idx, (expr, eval_count) in enumerate(no_error_exprs, start = 1):
                print(f"{idx}.  {expr.format()}")
                print(f"  Number of evaluations: {eval_count}")
            print()

    def __repr__(self) -> str:
        num_samples = len(self.samples)
        num_skipped = len(self.skipped_samples)
        
        exprs_summary = [
            {"expr": expr.format(), "mean_error": round(np.mean(evals).item(), 4) if evals else None} # TODO: 4 decimal places?
            for expr, evals in self.errors.items()
        ]
        
        return str({
            "sampled": num_samples,
            "skipped": num_skipped,
            "exprs": exprs_summary
        })
