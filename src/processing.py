# src/processing.py

import pandas as pd
from typing import Dict, Tuple
from . import config

def _merge_features(df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(df, features_df, on=['Store', 'Date', 'IsHoliday'], how='left')

def _merge_stores(df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(df, stores_df, on=['Store'], how='left')

def _process_dates_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    if config.DATE_COLUMN in df.columns:
        df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
        sort_keys = [col for col in [config.DATE_COLUMN, 'Store', 'Dept'] if col in df.columns]
        if sort_keys:
            df = df.sort_values(by=sort_keys).reset_index(drop=True)
    return df

def run_preprocessing(
    dataframes: Dict[str, pd.DataFrame],
    process_train: bool = True,
    process_test: bool = True,
    merge_features: bool = True,
    merge_stores: bool = True,
    drop_raw_components: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Main preprocessing function to orchestrate all merging and transformation.
    Applies a consistent pipeline with granular controls to the selected datasets.

    Args:
        dataframes (Dict): Raw dataframes from the data_loader.
        process_train (bool): If True, runs the pipeline on the train data.
        process_test (bool): If True, runs the pipeline on the test data.
        merge_features (bool): If True, merges the features data.
        merge_stores (bool): If True, merges the stores data.
        drop_raw_components (bool): If True, deletes raw dataframes after use.

    Returns:
        A dictionary containing the processed dataframes, keyed by name.
    """
    
    primary_to_process = []
    if process_train and "train" in dataframes:
        primary_to_process.append("train")
    if process_test and "test" in dataframes:
        primary_to_process.append("test")

    if not primary_to_process:
        print("Warning: No dataframes selected for processing.")
        return {}

    processed_dfs = {}
    
    for name in primary_to_process:
        df = dataframes[name].copy()

        if merge_features and "features" in dataframes:
            df = _merge_features(df, dataframes["features"])
        if merge_stores and "stores" in dataframes:
            df = _merge_stores(df, dataframes["stores"])
        
        df = _process_dates_and_sort(df)
        
        processed_dfs[name] = df

    if drop_raw_components:
        keys_to_drop = primary_to_process
        if merge_features:
             keys_to_drop.append("features")
        if merge_stores:
             keys_to_drop.append("stores")

        for key in keys_to_drop:
            if key in dataframes:
                del dataframes[key]
    
    return processed_dfs

def split_data(
    dataframe: pd.DataFrame,
    separate_target: bool = True,
    target_column: str = config.TARGET_COLUMN
):
    split_index = int(config.TRAIN_RATIO * len(dataframe))
    train_df = dataframe.iloc[:split_index]
    valid_df = dataframe.iloc[split_index:]

    if separate_target:
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_valid = valid_df.drop(columns=[target_column])
        y_valid = valid_df[target_column]
        return X_train, y_train, X_valid, y_valid
    return train_df, valid_df
