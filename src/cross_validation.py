import itertools
import copy
from typing import Any, Dict, Callable, Tuple
import pandas as pd

def manual_model_search(
    model: Any,
    param_grid: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    metric_func: Callable[[Any, Any], float] | Callable[[Any, Any], float],
    metric_kwargs: Dict[str, Any] = None,
    verbose: bool = True
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Performs manual grid search on an already-instantiated model.
    
    Args:
        model: An instance of the model (e.g., XGBRegressor(), RandomForestRegressor()).
        param_grid: Dict of param -> list of values.
        metric_func: Custom metric (takes y_true, y_pred, **metric_kwargs).
    
    Returns:
        best_model: Fitted model with best params.
        best_params: Params giving lowest validation error.
        best_score: Best metric score.
    """
    if metric_kwargs is None:
        metric_kwargs = {}

    best_score = float('inf')
    best_model = None
    best_params = None

    keys, values = zip(*param_grid.items())
    for combo in itertools.product(*values):
        trial_params = dict(zip(keys, combo))

        model_copy = copy.deepcopy(model)
        model_copy.set_params(**trial_params)

        try:
            model_copy.fit(X_train, y_train)
            preds = model_copy.predict(X_valid)
            score = metric_func(y_valid, preds, **metric_kwargs)

            if verbose:
                print(f"Params: {trial_params} -> Score: {score:.4f}")

            if score < best_score:
                best_score = score
                best_model = model_copy
                best_params = trial_params
        except Exception as e:
            if verbose:
                print(f"Params {trial_params} failed: {e}")
            continue

    return best_model, best_params, best_score
