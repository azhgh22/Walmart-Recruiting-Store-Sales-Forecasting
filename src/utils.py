import numpy as np
import pandas as pd
import wandb
import joblib

def wmae(y_true, y_pred, is_holiday):
    """
    Compute the Weighted Mean Absolute Error (WMAE).

    Parameters:
    - y_true: array-like of true values
    - y_pred: array-like of predicted values
    - is_holiday: array-like of bools or 0/1, where 1 indicates a holiday week

    Returns:
    - WMAE (float)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = np.where(np.array(is_holiday), 5, 1)

    abs_errors = np.abs(y_true - y_pred)
    weighted_errors = weights * abs_errors

    return weighted_errors.sum() / weights.sum()


def difference_series(series, lag=1):
    return series.diff(lag).dropna()


def log_to_wandb(model, train_score, val_score, config, 
                       run_name="run_00",
                       project_name="Walmart Recruiting - Store Sales Forecasting", 
                       artifact_name="pipeline", 
                       artifact_type="model", 
                       artifact_description=""):
    """
    Saves model, logs config, metrics, and artifact to wandb in one go.

    Args:
        model: Trained model object to save.
        train_score: Training score metric (e.g. WMAE).
        val_score: Validation score metric.
        config: dict, wandb config to log.
        run_name: wandb run name, also used as filename.
        project_name: wandb project name.
        artifact_name: name for the wandb artifact.
        artifact_type: artifact type string.
        artifact_description: description for artifact.
    """
    filename = f"{run_name}.pkl"
    joblib.dump(model, filename)
    
    wandb.init(project=project_name, name=run_name)
    wandb.config.update(config)
    
    wandb.log({
        'train_wmae': train_score,
        'val_wmae': val_score
    })
    
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=artifact_description
    )
    artifact.add_file(filename)
    wandb.log_artifact(artifact)
    
    wandb.finish()
