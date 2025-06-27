import pandas as pd
from typing import Optional, List, Dict, Tuple
from . import config

def load_raw_data(
    dataframes_to_load: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Loads specified raw data files from the paths defined in the config.

    Args:
        dataframes_to_load (Optional[List[str]]): 
            A list of the names of the dataframes to load (e.g., "train", "stores").
            If None, all available raw data files will be loaded. Defaults to None.

    Returns:
        A dictionary mapping the loaded dataframe names to their pandas DataFrames.
        
    Raises:
        ValueError: If a requested dataframe name is not valid.
    """

    AVAILABLE_DATAFRAMES = {
        "stores": config.STORES_PATH,
        "features": config.FEATURES_PATH,
        "train": config.TRAIN_PATH,
        "test": config.TEST_PATH,
        "sample_submission": config.SAMPLE_SUBMISSION_PATH
    }
    
    if dataframes_to_load is None:
        dataframes_to_load = list(AVAILABLE_DATAFRAMES.keys())
    else:
        for name in dataframes_to_load:
            if name not in AVAILABLE_DATAFRAMES:
                raise ValueError(
                    f"'{name}' is not a valid dataframe name. "
                    f"Choose from: {list(AVAILABLE_DATAFRAMES.keys())}"
                )
    loaded_dataframes = {}
    for name in dataframes_to_load:
        path = AVAILABLE_DATAFRAMES[name]
        loaded_dataframes[name] = pd.read_csv(path)
            
    print("Data loading complete.")
    return loaded_dataframes