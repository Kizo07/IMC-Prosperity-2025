import subprocess
import numpy as np
import os
import re
import pandas as pd  # Import pandas for DataFrame handling

"""
SETUP

- change grid_search_values to desired numbers below
- change backtest_day to desired round
- choose template_file name
    ensure that template file has variable "grid_search = " defined at the top
    replace desired variable with grid_search in your template code
- ensure products to track are correct

no logs will be saved. Run the code manually to view logs for specific parameters

The grid search results will be in a dataframe which can be explored to see which maximizes profit

"""
grid_search_values = np.arange(-1, 0, 0.01).round(2).tolist()
backtest_day = 0
template_file = "sample-program3.py"
products_to_track = ["KELP", "RAINFOREST_RESIN"]
results = []

""" main script """
def write_temp_version(base_file: str, param: float, i: int) -> str:
    with open(base_file, "r") as f:
        lines = f.readlines()

    temp_file = f"temp_program_{i}.py"
    with open(temp_file, "w") as f:
        for line in lines:
            if line.strip().startswith("grid_search ="):
                f.write(f"grid_search = {param}\n")
            else:
                f.write(line)
    return temp_file

# Run grid search
for i, param in enumerate(grid_search_values):
    temp_file = write_temp_version(template_file, param, i)

    try:
        process = subprocess.run(
            ["prosperity3bt", temp_file, str(backtest_day), "--no-out"],
            capture_output=True,
            text=True
        )
        output = process.stdout.strip()

        # Extract total profit
        total_match = re.search(r"Total profit:\s*([-\d,]+)", output)
        total_profit = int(total_match.group(1).replace(",", "")) if total_match else None

        # Extract per-product profits
        product_profits = {}
        for product in products_to_track:
            pattern = fr"{product}:\s*([-\d,]+)"
            match = re.search(pattern, output)
            if match:
                product_profits[product] = int(match.group(1).replace(",", ""))
            else:
                product_profits[product] = None

        # Append results as a dictionary
        results.append({
            "grid_search": param,
            "total_profit": total_profit,
            **product_profits  # Unpack product profits into the dictionary
        })

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Convert results to a Pandas DataFrame
df_results = pd.DataFrame(results)
print(df_results)