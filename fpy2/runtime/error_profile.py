from ..ir import Expr
from typing import Any, Literal
import numpy as np

Verbosity = Literal["minimal", "standard", "detailed"]

class ErrorProfile:
    """
    A class to store and analyze the errors in a set of sampled data, 
    including the skipped samples and their associated error statistics.
    """

    inputs: list[Any]

    skipped_inputs: list[Any]

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
        percent_skipped   = (num_skipped / num_samples) * 100 if num_samples > 0 else 0
        total_expressions = len(self.errors)
        
        print(f"Number of Sample points     : {num_samples}")
        print(f"Skipped sample points       : {num_skipped}")
        print(f"Percentage of points skipped: {percent_skipped:.2f}%")
        print(f"Number of expressions       : {total_expressions}\n")
        
        for idx, (expr, evaluations) in enumerate(self.errors.items(), start=1):
            print(f"Expression {idx}: {expr.format()}")
            print(f"  Number of evaluations: {len(evaluations)}")
            
            if evaluations:
                statistics = self._compute_statistics(evaluations, verbosity)
                
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
