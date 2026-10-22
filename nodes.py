from state import State
import pandas as pd

def loader_node(state: State) -> dict:
    filepath = state[filepath]
    df = pd.read_csv(filepath)

    return{'dataframe': df}

def profiler_node(state: State) -> dict:
    df = state['dataframe']
    shape = df.shape 
    size = df.size
    columns = df.columns.tolist()
    info = df.info()
    numerical_description = df.describe()
    categorical_description = df.describe(include="all")
    null_values = df.isnull().sum() 
    null_values_percentage = df.isnull().mean() * 100 
    duplicate_rows = df.duplicated().sum()
    Unique_values_per_column = df.nunique()
    variance = df.var(numeric_only=True)
    skewness = df.skew(numeric_only=True)
    correlation = df.corr(numeric_only=True)
    profile = {
        'shape': shape,
        'size': size,
        'all_columns': columns,
        'all_column_info': info,
        'numerical_column_description': numerical_description,
        'categorical_column_description': categorical_description,
        'total_null_values_per_column': null_values,
        'percentage_of_null_values_per_column': null_values_percentage,
        'total_no_of duplicated_rows': duplicate_rows,
        'unique values_per_column': Unique_values_per_column,
        'variance_of numerical_columns': variance,
        'skewness_of_numerical_columns': skewness,
        'correlation_among_all_numerical_columns': correlation
    }

    return{'data_profile': profile}

    