from langchain_core.messages import SystemMessage, HumanMessage
from typing import Literal, Optional
from pydantic import BaseModel, Field
from llm import gemini_llm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from registry import MODEL_REGISTRY
from helping_function import detect_outliers
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
from state import State

def train_test_split(state: State)-> dict:
    target = state['target_column']
    df = state['dataframe']
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return{
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

def select_model(state: State)-> dict:
    plan = state["plan"]

    selected_model = next(
        model.name
        for model in plan.candidate_models
    )

    return {"selected_model": selected_model}

def should_scale(state: State) -> dict:
    selected_models = state['selected_model']
    need_scaling = {}
    for model in selected_models:
        need_scaling['model'] = MODEL_REGISTRY["model"]["needs_scaling"]
    return {'should_scale': need_scaling}


def analyze_scaling_needs(state: State)-> dict:
    X_train = state['X_train']
    numerical_columns = X_train.select_dtypes(include="number")
    scaling_report = {}
    for col in numerical_columns.items():
        outlier = detect_outliers(col)
        scaling_report[f"scaling_report_of_{col}"] = {
            'unique_values': col.nunique(),
            'mean': col.mean(),
            'std': col.std(),
            'min': col.min(),
            'max': col.max(),
            'skewness': col.skew(),
            'outlier': outlier,
            'is_constant': col.nunique()<=1,
            'missing_percentage': round(float(col.isna().mean() * 100), 2),
            'is_binary': col.nunique() == 2 and set(col.dropna().unique()).issubset({0, 1})
        }

    return {'scaling_report': scaling_report}

def choose_scaler(state: State)-> dict:
    need_scale = state['should_scale']
    scaling_report = state['scaling_report']

    if (need_scale == False):
        pass



             


