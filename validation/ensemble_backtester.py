import itertools
from .full_backtester import FullBacktester
from .metrics import METRICS

class FullBacktesterEnsemble:
    def __init__(self, df, turnover_df, config_list):
        """
        Parameters:
          - df: Returns DataFrame covering the entire period.
          - turnover_df: Turnover DataFrame (aligned with df).
          - config_list: A list of configuration dictionaries. Each configuration dictionary
                         should include the keys:
                           "first_os", "window_length", "step_months", "anchored",
                           "risk_free_rate", "top_n", "max_corr", "max_columns",
                           "min_avg_trade", "metric_name"
        """
        self.df = df
        self.turnover_df = turnover_df
        self.config_list = config_list
        self.results = {}

    def run(self):
        for config in self.config_list:
            # Unpack parameters from the configuration.
            first_os = config["first_os"]
            window_length = config["window_length"]
            step_months = config["step_months"]
            anchored = config["anchored"]
            risk_free_rate = config["risk_free_rate"]
            top_n = config["top_n"]
            max_corr = config["max_corr"]
            max_columns = config["max_columns"]
            min_avg_trade = config["min_avg_trade"]
            metric_name = config["metric_name"]

            # Retrieve the metric function from METRICS.
            metric_func = METRICS.get(metric_name)
            if metric_func is None:
                raise ValueError(f"Metric '{metric_name}' is not defined in METRICS.")
            
            # Build a configuration key for display.
            config_key = f"{metric_name}_WL{window_length}_Anchored{anchored}_Step{step_months}"
            print(f"Running configuration: {config_key}")
            
            # Create and run the FullBacktester for this configuration.
            fb = FullBacktester(
                self.df, self.turnover_df,
                first_os, window_length, step_months, anchored,
                risk_free_rate, metric_func,
                top_n, max_corr, max_columns, min_avg_trade
            )
            fb.run_in_sample()
            fb.run_oos()
            agg = fb.aggregate_oos()
            self.results[config_key] = agg
        return self.results

def generate_config_list(config_grid):
    """
    Given a dictionary where each key maps to a list of candidate values,
    generate a list of configuration dictionaries using the Cartesian product.
    """
    config_list = [dict(zip(config_grid.keys(), values))
                   for values in itertools.product(*config_grid.values())]
    return config_list

def print_available_metrics():
    """
    Print the available metrics from the METRICS dictionary.
    """
    print("Available metrics:")
    for key in METRICS.keys():
        print(f" - {key}")

if __name__ == "__main__":
    # If run directly, print available metrics.
    print_available_metrics()
