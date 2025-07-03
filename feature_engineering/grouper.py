import pandas as pd

def group_and_aggregate(
    X: pd.DataFrame,
    y: pd.Series = None,
    groupby_cols=None,
    y_aggfunc='sum'
):
    """
    Groups a DataFrame `X` by specified columns and processes it as follows:

    - Retains only the columns whose values are constant within each group.
    - From each group, takes the row with the lowest original index.
    - Resets the index of the resulting DataFrame.
    - If `y` is provided, aggregates it according to the same grouping and returns the result.

    Parameters:
        X (pd.DataFrame): The input feature DataFrame.
        y (pd.Series or None): The target variable to aggregate. Optional.
        groupby_cols (list of str): List of column names to group by.
        y_aggfunc (str or callable): Aggregation function to apply to y (e.g., 'sum', 'mean').

    Returns:
        Tuple:
            - pd.DataFrame: Grouped DataFrame with constant columns retained and reset index.
            - pd.Series or None: Aggregated target values aligned with grouped rows, or None if `y` is not provided.

    Raises:
        ValueError: If `groupby_cols` is not provided.
    """
    if groupby_cols is None:
        raise ValueError("groupby_cols must be provided")

    grouped = X.groupby(groupby_cols)
    constant_cols = []
    for col in X.columns:
        if col in groupby_cols:
            continue
        if (grouped[col].nunique() <= 1).all():
            constant_cols.append(col)
    first_indices = grouped.apply(lambda g: g.index.min()).values
    selected_cols = groupby_cols + constant_cols
    X_result = X.loc[first_indices, selected_cols].reset_index(drop=True)

    if y is not None:
        df_y = X[groupby_cols].copy()
        df_y['__target__'] = y.values
        y_grouped = df_y.groupby(groupby_cols)['__target__'].agg(y_aggfunc)
        y_result = X_result[groupby_cols].apply(lambda row: y_grouped.loc[tuple(row)], axis=1)
        return X_result, y_result.reset_index(drop=True)
    else:
        return X_result, None
