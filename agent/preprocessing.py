from langchain_core.messages import SystemMessage, HumanMessage
from typing import Literal, Optional
from pydantic import BaseModel, Field
from llm import gemini_llm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from registry import MODEL_REGISTRY
from helping_function import detect_outliers, percentage_of_outliers, remove_rows_with_missing_target, replace_fake_missing_values_with_nan
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
from state import State
import pandas as pd

def choosing_target_column(state: State) ->dict:
    pass

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

#********************************************************************DATA CLEANING**********************************************************************


def data_cleaning(state: State) -> dict:
    X_train = state['X_train']
    y_train = state['y_train']

    X_train = replace_fake_missing_values_with_nan(X_train)
    X_train, y_train = remove_rows_with_missing_target(X_train, y_train)
    





    



# class DataCleaningPlan(BaseModel):
#     columns_to_drop: Optional[list[str]] = Field(default_factory=list)
#     columns_to_impute: Optional[list[str]] = Field(default_factory=list)
#     columns_to_encode: Optional[list[str]] = Field(default_factory=list)
#     columns_to_scale: Optional[list[str]] = Field(default_factory=list)




# def data_cleaning(state: State) -> dict:
#     X_train = state['X_train']
#     column_registry = state['column_registry']

#     for col, info in column_registry.items():
#         if info["semantic_type"] == "id":
#             X_train.drop(columns=[col], inplace=True)
#         elif info["semantic_type"] == "constant":
#             X_train.drop(columns=[col], inplace=True)
        



#**********************************************************************IMPUTATION********************************************************************

def analyze_missing_values(state: State) -> dict:

    X_train = state["X_train"]

    column_types = {}
    column_registry = state["column_registry"]
    for col, info in column_registry.items():
        semantic_type = info["semantic_type"]
        column_types[col] = semantic_type

    missing_report = {}

    for col in X_train.columns:
        n_missing = int(X_train[col].isnull().sum())
        if n_missing == 0:
            continue

        non_missing_values = X_train[col].dropna()
        kind = column_types.get(col, "unknown")
        info = {
            "kind": kind,
            "n_missing": n_missing,
            "missing_percentage": round(float(X_train[col].isnull().mean() * 100), 2),
            "unique_values": int(non_missing_values.nunique()),
        }
        if kind.startswith("numeric"):
            info["skewness"] = float(non_missing_values.skew()) if len(non_missing_values) > 2 else 0.0
            info["percentage_of_outlier"] = percentage_of_outliers(non_missing_values)

        missing_report[col] = info

    return {"missing_report": missing_report}

def choose_imputer(state: State) -> dict:
    report = state["missing_report"]
    model_name = state["selected_model"]
    handles_nan = MODEL_REGISTRY[model_name].get("native_missing", False)

    column_registry = state["column_registry"]

    numeric_columns = [
        col for col, info in column_registry.items()
        if info["semantic_type"].startswith("numeric")
    ]

    n_rows = len(state["X_train"])
    n_numeric = len(numeric_columns)

    DROP_PCT = 60
    INDICATOR_PCT = 5
    KNN_MAX_PCT = 30

    knn_ok = n_numeric >= 3 and n_rows <= 20_000

    plan, reasoning = {}, {}

    for col, r in report.items():
        pct = r["missing_percentage"]
        kind = r["kind"]
        is_numeric = kind.startswith("numeric")

        if pct > DROP_PCT:
            plan[col] = {"strategy": "drop", "add_indicator": False}
            reasoning[col] = f"{pct}% missing, too much to impute reliably"
            continue

        if handles_nan and is_numeric:
            plan[col] = {"strategy": "passthrough", "add_indicator": False}
            reasoning[col] = f"{model_name} handles NaN natively"
            continue

        indicator = pct >= INDICATOR_PCT

        if is_numeric:
            skewed = abs(r["skewness"]) >= 1 or r["percentage_of_outlier"] > 5
            if pct <= KNN_MAX_PCT and pct >= INDICATOR_PCT and knn_ok:
                strategy = "knn"
                why = f"{pct}% missing, KNN uses neighbouring rows"
            elif skewed:
                strategy = "median"
                why = f"Skewed or outlier-heavy, median is robust ({pct}% missing)"
            else:
                strategy = "mean"
                why = f"Roughly symmetric, mean is fine ({pct}% missing)"

        elif kind == "binary" or r["unique_values"] == 2:
            strategy, why = "most_frequent", f"Binary column, mode ({pct}% missing)"

        elif kind in ("categorical_nominal", "categorical_ordinal"):
            if pct < INDICATOR_PCT:
                strategy, why = "most_frequent", f"Few missing ({pct}%), mode"
            else:
                strategy, why = "constant", f"{pct}% missing, treat as its own 'Missing' category"
                indicator = False  
        else:
            strategy, why = "most_frequent", f"Unknown type, safe default ({pct}% missing)"

        plan[col] = {"strategy": strategy, "add_indicator": indicator}
        reasoning[col] = why

    return {"imputer_plan": plan, "imputer_reasoning": reasoning}

#*******************************************************************ENCODING*******************************************************************

def choose_encoder(state:State) ->dict:
    X_train = state['X_train']
    column_info = {}
    plan = {}
    reasoning = {}
    column_registry = state["column_registry"]
    for col, info in column_registry.items():
        column_info[col] = {
            'semantic_type': info["semantic_type"],
            'ordinal_order': info['ordinal_order'],
            'role': info['role']
        }
        
    for col in X_train:

        if column_info[col]['semantic_type'] == 'id':
            plan[col] = {'strategy': 'drop', 'ordinal_order': False}
            reasoning[col] = {"ID-like column, no predictive value"}

        elif column_info[col]['semantic_type'].startswith('numeric') or column_info[col]['semantic_type'] == 'binary' and pd.api.types.is_numeric_dtype(X_train[col]):
            plan[col] = None
            reasoning[col] = {'Numeric Column does not need to be Encoded'}

        elif column_info[col]['semantic_type'] == "categorical_nominal":
            plan[col] = {'strategy': 'one_hot_encoding', 'ordinal_order': False}
            reasoning[col] = {'Nominal categorical features have no inherent ordering, so One-Hot Encoding represents each category independently without introducing artificial ordinal relationships.'}

        elif column_info[col]['semantic_type'] == 'categorical_ordinal':
            plan[col] = {'strategy': 'ordinal encoding', 'ordinal_order': column_info[col]['ordinal_order']}
            reasoning[col] = {'Ordinal categorical features have an inherent order or ranking. Ordinal Encoding preserves this ordering by assigning numerical values according to the category hierarchy.'}

        elif column_info[col]['semantic_type'] == 'constant':
            plan[col] = {'strategy': 'drop', 'ordinal_order': False}
            reasoning[col] = {'Constant features contain only one unique value and therefore provide no discriminative information to the model. Dropping them reduces unnecessary features without losing useful information.'}

        elif column_info[col]['semantic_type'] == ['text' or 'datetime']:
            plan[col] = None
            reasoning[col] = {'Text and datetime features require specialized preprocessing rather than conventional categorical encoding. Text features may require TF-IDF vectorization, while datetime features can be extracted into meaningful components such as year, month, day, and hour.'}

        else:
            plan[col] = None
            reasoning[col] = None

    return {'encoding_plan': plan, 'encoding_reasoning': reasoning}

    
def encoder_for_target_column(state: State) ->dict:
    pass
    

#********************************************************************SCALING**********************************************************************            

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




