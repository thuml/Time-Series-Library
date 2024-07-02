import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List
from pathlib import Path

def load_metrics(name: str, leadtime: int=96) -> pd.DataFrame:
    """
    Load metrics for a given model name and lead time.

    Args:
        name (str): The name of the model.
        leadtime (int, optional): The lead time. Defaults to 96.

    Returns:
        pd.DataFrame: A DataFrame containing the metrics.

    Raises:
        ValueError: If an invalid model name is provided.
    """

    idx_lst, rmse_lst, mape_lst = [], [], []

    if name == "lstm" or name == "lstm_shuffle" or name == "lstm_wrf":
        for l in range(1, leadtime+1):
            if name == "lstm":
                data_dir = Path(f"../exp_results/formosa_19_20_LSTM_WS_p{l}")
            elif name == "lstm_shuffle":
                data_dir = Path(f"../exp_results/formosa_19_20_LSTM_WS_p{l}_shffule")
            elif name == "lstm_wrf":
                data_dir = Path(f"../exp_results/formosa_wrf_19_20_LSTM_WS_p{l}")
            with open(data_dir/"result.json") as f:
                results = json.load(f)
            idx_lst.append(l)
            rmse_lst.append(results["test_rmse"])
            mape_lst.append(results["test_mape"])
        df = pd.DataFrame({
            "leadtime": idx_lst, "rmse": rmse_lst, "mape": mape_lst
        })
    else:
        df = pd.read_csv(f"{name}_rmse_mape.csv")
        df.loc[:, "leadtime"] = df["leadtime"] + 1
        df.loc[:, "mape"] = df["mape"] * 100

    return df

def plot_metrics(df: pd.DataFrame, metric: str, models: List[str]) -> plt.Figure:
    """
    Plots the specified metric for different models over time.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        metric (str): The metric to plot.
        models (List[str]): The list of models to include in the plot.

    Returns:
        plt.Figure: The matplotlib figure object representing the plot.
    """
    fig = px.scatter(
        df, x=df.index, y=models, 
        title=metric, labels={'x': 'Time', 'value': metric},
    )
    return fig

def load_true(name: str):
    data_dir = Path("../data_for_model")
    df = pd.read_csv(data_dir/f"{name}.csv")
    df = df[["datetime", "WS_90"]]
    df = df.rename(columns={"WS_90": "true"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    index_df = pd.read_csv('pred_index.csv')
    df = df[df.index.isin(index_df['date'])]
    return df


'''Read Data'''
def load_curve(dir: str, npy_name: str, prediction_length: int=96):
    """
    Load the wind speed curve data from a numpy file and return it as a DataFrame.

    Args:
        npy_name (str): The name of the numpy file to load.
        prediction_length (int, optional): The length of each prediction in the numpy file. Defaults to 96.

    Returns:
        pd.DataFrame: The wind speed curve data as a DataFrame.
    """
    index_df = pd.read_csv('../results/test_index.csv')
    index_df['date'] = pd.to_datetime(index_df['date'])
    
    npy = np.load(f"../results/{dir}/{npy_name}.npy").reshape(-1, prediction_length)
    df = pd.DataFrame(npy).join(index_df)
    df = df.rename(columns={i: f"{npy_name}_p{i+1}" for i in range(prediction_length)})
    df = df.dropna().set_index("date")

    return df

def load_lstm_curve(attr: str="", prediction_length: int=96) -> pd.DataFrame:
    """
    Load LSTM curve data for wind speed prediction.

    Parameters:
        prediction_length (int): The length of the prediction.

    Returns:
        pd.DataFrame: The concatenated dataframe containing the LSTM curve data.
    """
    attr = attr if attr == "" else "_" + attr
    df_lst = []
    for l in range(1, prediction_length+1):
        data_dir = Path(f"../exp_results/formosa{attr}_19_20_LSTM_WS_p{l}")
        df = pd.read_csv(data_dir/"pred.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.rename(columns={col: f"lstm{attr}_pred_p{l}" for col in df.columns})
        df_lst.append(df)

    df = pd.concat(df_lst, axis=1)

    return df

def plot_curve_by_leadtime(df: pd.DataFrame, leadtime: int, labels: List[str]) -> plt.Figure:
    """
    Plots the wind speed curves for a given lead time.

    Args:
        df (pd.DataFrame): The DataFrame containing the wind speed data.
        leadtime (int): The lead time for which the curves are plotted.
        labels (List[str]): The labels for the wind speed curves.

    Returns:
        plt.Figure: The matplotlib Figure object containing the plotted curves.
    """
    curves = [f'{l}_p{leadtime}' for l in labels] + ["true"]
    fig = px.line(
        df, x=df.index, y=curves, 
        title=f"Lead Time {leadtime}", labels={'x': 'Time', 'value': 'Wind Speed'},
    )
    return fig

def plot_curve_by_model(df: pd.DataFrame, model: str, leadtime: int=96) -> plt.Figure:
    """
    Plots the curves for a given model's predictions and the true wind speed values.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        model (str): The name of the model.
        leadtime (int, optional): The number of lead times to plot. Defaults to 96.

    Returns:
        plt.Figure: The matplotlib Figure object containing the plot.
    """
    
    curves = [f'{model}_p{l}' for l in range(1, leadtime+1) ] + ["true_p1"]
    fig = px.line(
        df, x=df.index, y=curves, 
        title=f"{model}", labels={'x': 'Time', 'value': 'Wind Speed'},
    )
    return fig
