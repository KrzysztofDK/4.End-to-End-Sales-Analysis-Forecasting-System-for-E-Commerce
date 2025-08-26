"""Module with utility functions."""

import os
import sys

import matplotlib.pyplot as plt

from src.logger import logging
from src.exception import CustomException


def save_forecast_and_plot(
    prophet_model, history_df, periods_list: list[int, int, int] = [30, 60, 90]
):
    """
    Creates a forecast from the Prophet model, saves the data to CSV and the chart to PNG.

    Args:
        prophet_model: Trained Prophet model.
        history_df (pd.DataFrame): Historical data with columns ["ds", "y"].
        periods (list[int, int, int], optional): Number of days to predict. Default is 30, 60, and 90 as a list.
    """

    logging.info("Function to plot and save forecast has started.")

    try:
        saved_paths = []
        for periods in periods_list:
            future = prophet_model.make_future_dataframe(periods=periods, freq="D")
            forecast = prophet_model.predict(future)

            csv_path = os.path.join("reports", f"forecast_{periods}d.csv")
            forecast.tail(periods).to_csv(csv_path, index=False)

            plt.figure(figsize=(12, 6))
            plt.plot(
                history_df["ds"], history_df["y"], label="Historical data", color="blue"
            )
            plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="red")
            plt.fill_between(
                forecast["ds"],
                forecast["yhat_lower"],
                forecast["yhat_upper"],
                color="red",
                alpha=0.2,
                label="Confidence interval",
            )

            plt.title(f"Revenue per Day – forecast for {periods} days")
            plt.xlabel("Date")
            plt.ylabel("Revenue")
            plt.legend()
            plt.grid(True)

            png_path = os.path.join("reports", f"forecast_plot_{periods}d.png")
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close()

            saved_paths.append((csv_path, png_path))

    except Exception as e:
        logging.info("Function to plot and save forecast has encountered a problem.")
        raise CustomException(e, sys) from e
