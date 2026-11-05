from langchain_core.messages import SystemMessage, HumanMessage
from typing import Literal, Optional
from pydantic import BaseModel, Field
from llm import gemini_llm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from registry import MODEL_REGISTRY
from helping_function import detect_outliers, percentage_of_outliers
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
        if model.role == "primary"
    )

    return {"selected_model": selected_model}



# def handling_missing_value(state: State)-> dict:
#     X_train = state['X_train']

#     missing_value_per_column = {}
#     missing_value_per_column_in_percentage = {}
#     for col in X_train:
#         missing_value_per_column[col] = X_train[col].isnull().sum()
#         missing_value_per_column_in_percentage[col] = X_train[col].isnull().mean().mul(100).round(2)

#     missing_value_plan = {}
#     missing_value_reasoning = {}

#     for col, pct in missing_value_per_column_in_percentage.items():
#         if pct >= 60:
#             missing_value_plan[col] : 's'
            

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
        outlier_percentage = percentage_of_outliers(col)
        scaling_report[f"scaling_report_of_{col}"] = {
            'unique_values': col.nunique(),
            'mean': col.mean(),
            'std': col.std(),
            'min': col.min(),
            'max': col.max(),
            'skewness': col.skew(),
            'outlier': outlier,
            'percentage_of_outlier': outlier_percentage,
            'is_constant': col.nunique()<=1,
            'missing_percentage': round(float(col.isna().mean() * 100), 2),
            'is_binary': col.nunique() == 2 and set(col.dropna().unique()).issubset({0, 1})
        }

    return {'scaling_report': scaling_report}

SCALER_CLASSES = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
    "MaxAbsScaler": MaxAbsScaler,
    "PowerTransformer": lambda: PowerTransformer(method="yeo-johnson"),
    "QuantileTransformer": lambda: QuantileTransformer(output_distribution="normal"),
}

def choose_scaler(state: State)-> dict:
    need_scale = state['should_scale']
    scaling_report = state['scaling_report']

    needs = any(need_scale.values()) if isinstance(need_scale, dict) else bool(need_scale)
    if not needs:
        return {
            "scaler_plan": {},
            "scaler_reasoning": {"all": "Selected model does not need scaling"},
        }

    plan, reasoning = {}, {}

    for col, report in scaling_report.items():
        outlier_percentage = report["percentage_of_outlier"]
        skew = report["skewness"]
        skew = abs(skew) if skew == skew else 0 
        if report["is_constant"]:
            plan[col] = None
            reasoning[col] = "Constant column, carries no information; consider dropping it"
        elif report["is_binary"]:
            plan[col] = None
            reasoning[col] = "Binary 0/1 column, already on a common scale"
        elif skew >= 2 and report["unique_values"] > 20:
            plan[col] = "QuantileTransformer"
            reasoning[col] = f"Very skewed (|skew|={skew:.2f})"
        elif skew >= 1:
            plan[col] = "PowerTransformer"
            reasoning[col] = f"Moderately skewed (|skew|={skew:.2f})"
        elif outlier_percentage > 5:
            plan[col] = "RobustScaler"
            reasoning[col] = f"{outlier_percentage:.1f}% outliers"
        elif report["min"] >= 0 and outlier_percentage <= 1 and report["unique_values"] <= 20:
            plan[col] = "MinMaxScaler"
            reasoning[col] = "Non-negative, low cardinality, no outliers"
        else:
            plan[col] = "StandardScaler"
            reasoning[col] = "Roughly symmetric with few outliers"

        return {"scaler_plan": plan, "scaler_reasoning": reasoning}


# def choose_encoder(state: State)-> dict:
#     X_train = state['X_train']
#     categorical_column = X_train.select_dtypes(include=["object", "category", "bool"])
#     for col in categorical_column:




             


