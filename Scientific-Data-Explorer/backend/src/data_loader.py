import pandas as pd
import numpy as np
from pathlib import Path


def load_dataset(file_path):
    '''Opens the CSV file from disk into a Pandas DataFrame.
    Raises a clean FileNotFoundError if the path is invalid.'''
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError("Dataset not found at path.")

    return pd.read_csv(path)


def get_dataset_summary(df):
    '''Extracts structural metadata: total row count, column count,
    column data types, missing value counts, and estimated memory usage in RAM.'''
    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'total_nulls': int(df.isnull().sum().sum()),
        'dtypes': df.dtypes.to_dict(),
        'memory_mb': round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
    }


def clean_data(df, fill_value=0.0):
    '''Fills missing values with a default value (default is 0.0 for sparse conversion).'''
    return df.fillna(fill_value)


def extract_numeric_matrix(df):
    '''Filters out text/categorical columns and extracts strictly numeric
    columns as a 2D NumPy array for downstream linear algebra.'''
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError('No numeric columns found in the dataset to build a matrix.')

    return numeric_df.to_numpy(dtype=np.float64)
