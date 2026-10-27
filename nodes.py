from state import State
import pandas as pd
from helping_function import detect_outliers, find_potential_ids, find_text_columns
from pandas.api.types import is_datetime64_any_dtype

def loader_node(state: State) -> dict:
    filepath = state[filepath]
    df = pd.read_csv(filepath)

    return{'dataframe': df}

def profiler_node(state: State) -> dict:
    df = state['dataframe']
    shape = df.shape
    no_of_rows = shape[0]
    no_of_columns = shape[1] 
    columns = df.columns.tolist()
    potential_id_columns = find_potential_ids(df)
    text_columns = find_text_columns(df)
    numerical_column = df.select_dtypes(include="number")
    datetime_columns = [
            col
            for col in df.columns
            if is_datetime64_any_dtype(df[col])
            ]
    
    size = df.size
    
    info = {
        "dtypes": df.dtypes.astype(str).to_dict(),
        "non_null_counts": df.notna().sum().to_dict(),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum())
    }
    numerical_description = df.describe().to_dict()
    categorical_description = df.describe(
        include = ["object", "category", "bool"]
    ).to_dict()
    null_values = df.isnull().sum().to_dict()

    null_values_percentage = (
        df.isnull().mean().mul(100).round(2).to_dict()
    )

    constant_cols = df.columns[df.nunique(dropna=True) <= 1].tolist()

    duplicate_rows = df.duplicated().sum()
    Unique_values_per_column = df.nunique().to_dict()
    variance = df.var(numeric_only=True).to_dict()
    skewness = df.skew(numeric_only=True).to_dict()
    correlation = df.corr(numeric_only=True).to_dict()

    outlier = {}

    for col in df.select_dtypes(include="number").columns:
        outlier[col] = detect_outliers(df[col])

    profile = {
        'shape': shape,
        'no_of_rows': no_of_rows,
        'no_of_columns': no_of_columns,
        'size': size,
        'all_columns': columns,
        'potential_id_columns':potential_id_columns,
        'text_columns': text_columns,
        'numerical_columns': numerical_column,
        'datetime_columns': datetime_columns,
        'all_column_info': info,
        'numerical_column_description': numerical_description,
        'categorical_column_description': categorical_description,
        'total_missing_values_per_column': null_values,
        'percentage_of_missing_values_per_column': null_values_percentage,
        'constant_columns': constant_cols,
        'total_no_of duplicated_rows': duplicate_rows,
        'unique values_per_column': Unique_values_per_column,
        'variance_of numerical_columns': variance,
        'skewness_of_numerical_columns': skewness,
        'correlation_among_all_numerical_columns': correlation,
        'Outlier': outlier
    }

    return{'data_profile': profile}


    
     