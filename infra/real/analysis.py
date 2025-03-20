import json

with open("experiment_result.json", "r") as f:
    data = json.load(f)


def calculate_error_metrics(data, k_values=[0, 1, 5, 10]):
    # Reorganize the structure to have k as the first key and system as the second key
    metrics = {
        k: {
            s: {"TP": 0, "FN": 0, "FP": 0, "TN": 0, "UP": 0, "UN": 0, "F": 0}
            for s in ["0", "1", "2", "3"]
        }
        for k in k_values
    }

    # Iterate through each function
    for function_name, function_data in data.items():
        # Iterate through each system (0, 1, 2, 3)
        real_evaluation = None
        for system_key, system_data in function_data.items():
            # Save system 0 (real evaluation) for later comparison
            if system_key == "0":
                real_evaluation = system_data
            
            if not system_data["exprs"]:
                for k in k_values:
                    metrics[k][system_key]["F"] += 1

            # For each expr in system data
            for expr_idx, expr_data in enumerate(system_data["exprs"]):
                mean_real = (
                    real_evaluation["exprs"][expr_idx]["stats"]["Mean"]
                    if real_evaluation["exprs"]
                    else None
                )
                mean_arbitrary = expr_data["stats"]["Mean"]

                # Compare to thresholds k
                for k in k_values:
                    # Rival fails
                    if mean_real == None:
                        if mean_arbitrary > k:
                            metrics[k][system_key]["UP"] += 1
                        else:
                            metrics[k][system_key]["UN"] += 1
                    else:
                        # Error detected by both
                        if mean_real > k and mean_arbitrary > k:
                            metrics[k][system_key]["TP"] += 1
                        # Error detected by real evaluation but not arbitrary precision
                        elif mean_real > k and mean_arbitrary <= k:
                            metrics[k][system_key]["FN"] += 1
                        # Error detected by arbitrary precision but not real evaluation
                        elif mean_real <= k and mean_arbitrary > k:
                            metrics[k][system_key]["FP"] += 1
                        else:
                            metrics[k][system_key]["TN"] += 1

    return metrics


error_metrics = calculate_error_metrics(data)
with open("analysis_result.json", "w") as f:
    json.dump(error_metrics, f, indent=4)
