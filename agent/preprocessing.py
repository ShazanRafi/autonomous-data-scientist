from agent.planner import cleaning_step_to_function_kwargs
from llm import gemini_llm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from registry import MODEL_REGISTRY
from helping_function import detect_outliers, percentage_of_outliers
from data_cleaning_functions import *
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
from state import State
import pandas as pd
import re

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
#********************************************************************REMOVING USELESS COLUMNS*******************************************************************

def drop_columns(state: State) -> dict:

    plan = state["plan"]
    decisions = plan.columns_to_drop

    target_name = state.get("target_column")
    X_train = state["X_train"]

    skipped = []
    drop_log = {}

    for d in decisions:
        col = d.column_name

        if col not in X_train.columns:
            skipped.append({"column": col, "reason": "not found in X_train (possibly hallucinated or already dropped)"})
            continue
        if col == target_name:
            skipped.append({"column": col, "reason": "refused: this is the target column"})
            continue
        if col in drop_log:
            continue 

        drop_log[col] = {
            "drop_reason": d.drop_reason,
            "evidence_from_profile": d.evidence_from_profile,
            "explanation": d.explanation,
        }

    input_columns = [c for c in X_train.columns if c not in drop_log]

    return {
        "X_train": X_train[input_columns],
        "input_columns_after_dropping": input_columns,
        "column_registry": {c: i for c, i in state["column_registry"].items() if c in input_columns},
        "dropped_columns": drop_log,
        "drop_skipped": skipped,
    }

#********************************************************************DATA CLEANING**********************************************************************


CLEANING_FUNCTIONS = {
    "replace_fake_missing_values_with_nan": replace_fake_missing_values_with_nan,
    "remove_rows_with_missing_target": remove_rows_with_missing_target,
    "remove_rows_that_are_completely_empty": remove_rows_that_are_completely_empty,
    "replace_infinite_values_with_nan": replace_infinite_values_with_nan,

    "trim_whitespace": trim_whitespace,
    "standardize_text_case": standardize_text_case,
    "fix_unicode_and_encoding_issues": fix_unicode_and_encoding_issues,
    "remove_special_characters": remove_special_characters,
    "merge_spelling_variants_of_same_category": merge_spelling_variants_of_same_category,
    "standardize_boolean_labels": standardize_boolean_labels,

    "convert_text_to_numeric": convert_text_to_numeric,
    "convert_percentage_text_to_number": convert_percentage_text_to_number,
    "convert_text_to_datetime": convert_text_to_datetime,
    "convert_numeric_codes_to_categorical": convert_numeric_codes_to_categorical,
    "convert_to_correct_dtype": convert_to_correct_dtype,
    "split_combined_column_into_parts": split_combined_column_into_parts,

    "convert_units_to_common_unit": convert_units_to_common_unit,
    "fix_impossible_values": fix_impossible_values,
    "fix_wrong_sign_values": fix_wrong_sign_values,
    "cap_physically_impossible_extremes": cap_physically_impossible_extremes,

    "remove_exact_duplicate_rows": remove_exact_duplicate_rows,
    "remove_duplicate_rows_by_key_columns": remove_duplicate_rows_by_key_columns,
}


def data_cleaning(state: State) -> dict:

    X_train = state["X_train"].copy()
    y_train = state["y_train"].copy()

    cleaning_reports = []

    for step in state["plan"].cleaning_steps:

        kwargs = cleaning_step_to_function_kwargs(step)
        cleaning_function = CLEANING_FUNCTIONS[step.cleaning_action]

        if step.cleaning_action in {
            "remove_rows_with_missing_target",
            "remove_rows_that_are_completely_empty"
        }:
            X_train, y_train = cleaning_function(
                X_train, y_train, **kwargs
            )

        elif step.cleaning_action == "remove_duplicate_rows_by_key_columns":
            X_train, report = cleaning_function(X_train, **kwargs)
            cleaning_reports.append(report)

        else:
            X_train = cleaning_function(X_train, **kwargs)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "cleaning_reports": cleaning_reports
    }


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

#********************************************************************ADDING MISSING INDICATOR COLUMNS**********************************************************************

def add_missing_indicator_column(state: State) -> dict:
    X_train: pd.DataFrame = state["X_train"].copy()

    plan = state.get("plan")
    steps = plan.preprocessing_steps if plan else []
    
    requested_cols: list[str] = []
    for step in steps:
        action = step.preprocessing_action
        affected_columns = step.columns_affected or []
        if action == "add_missing_indicator_column":
            for affected_col in affected_columns:
                if affected_col not in requested_cols:
                    requested_cols.append(affected_col)
            

    added, skipped = [], {}

    for col in requested_cols:
        if col not in X_train.columns:
            skipped[col] = "column not found in X_train(dropped earlier?)"
            continue

        n_missing_train = int(X_train[col].isnull().sum())
        if n_missing_train == 0:
            skipped[col] = "no missing values in train, indicator would be a constant 0"
            continue
        if n_missing_train == len(X_train):
            skipped[col] = "all values missing in train, indicator would be a constant 1"
            continue

        new_col = f"{col}_was_missing"
        if new_col in X_train.columns:
            skipped[col] = f"'{new_col}' already exists"
            continue

        X_train[new_col] = X_train[col].isnull().astype("int8")

        added.append({
            "source_column": col,
            "indicator_column": new_col,
            "train_missing_count": n_missing_train,
            "train_missing_pct": round(100 * n_missing_train / len(X_train), 2),
        })


    return {
        "X_train": X_train,
        "added_missing_indicator_columns": added,
        "skipped_missing_indicator_columns": skipped
    }


#********************************************************************GROUPING RARE CATEGORIES**********************************************************************

def group_rare_categories(state: State) -> dict:
    X_train: pd.DataFrame = state["X_train"].copy()

    plan = state.get("plan")
    steps = plan.preprocessing_steps if plan else []

    registry = state.get("column_registry")
    semantic_type = {c.name: c.semantic_type for c in registry.columns} if registry else {}

    rare_category_map: dict[str, dict] = {}
    report, skipped = [], {}

    for step in steps:
        if step.preprocessing_action != "group_rare_categories":
            continue

        min_freq = step.rare_category_min_frequency or 0.01
        label = step.rare_category_label or "Other"

        for col in step.columns_affected or []:
            if col in rare_category_map or col in skipped:
                continue  

            if col not in X_train.columns:
                skipped[col] = "column not found in X_train (dropped earlier?)"
                continue

            if semantic_type.get(col) != "categorical_nominal":
                skipped[col] = f"semantic type is '{semantic_type.get(col, 'unknown')}', only categorical_nominal is grouped"
                continue

            series = X_train[col].astype("object")
            counts = series.dropna().value_counts()  
            if counts.empty:
                skipped[col] = "column is entirely missing in train"
                continue

            threshold = min_freq * counts.sum()
            rare = counts[counts < threshold].index.tolist()
            frequent = counts[counts >= threshold].index.tolist()

            if len(rare) < 2:
                skipped[col] = f"fewer than 2 rare categories at min_frequency={min_freq}"
                continue
            if not frequent:
                skipped[col] = "every category is rare at this threshold"
                continue

            final_label = label
            while final_label in frequent:      
                final_label += "_grouped"

            X_train[col] = series.where(~series.isin(rare), final_label)

            rare_category_map[col] = {
                "frequent_categories": frequent,
                "label": final_label,
                "min_frequency": min_freq,
            }
            report.append({
                "column": col,
                "n_categories_before": int(len(counts)),
                "n_categories_after": int(len(frequent) + 1),
                "n_rare_grouped": int(len(rare)),
                "rows_affected_train": int(counts[rare].sum()),
            })

    return {
        "X_train": X_train,
        "rare_category_map": rare_category_map,
        "grouped_rare_categories_report": report,
        "skipped_group_rare_categories": skipped,
    }


#********************************************************************TRANSFORMING NUMERIC DISTRIBUTION**********************************************************************

SUPPORTED_DISTRIBUTION_METHODS = {"log1p", "sqrt", "yeo_johnson"}
NON_NEGATIVE_ONLY_METHODS = {"log1p", "sqrt"}
TRANSFORMABLE_TYPES = {"numeric_continuous", "numeric_discrete"}


def transform_numeric_distribution(state: State) -> dict:
    X_train: pd.DataFrame = state["X_train"].copy()

    plan = state.get("plan")
    steps = plan.preprocessing_steps if plan else []

    registry = state.get("column_registry")
    semantic_type = {c.name: c.semantic_type for c in registry.columns} if registry else {}

    fitted_transforms: dict[str, dict] = {}  
    report, skipped = [], {}

    for step in steps:
        if step.preprocessing_action != "transform_numeric_distribution":
            continue

        method = step.method.strip().lower()

        for col in step.columns_affected or []:
            if col in fitted_transforms or col in skipped:
                continue  # first step for a column wins

            if method not in SUPPORTED_DISTRIBUTION_METHODS:
                skipped[col] = f"unsupported method '{step.method}', supported: {sorted(SUPPORTED_DISTRIBUTION_METHODS)}"
                continue

            if col not in X_train.columns:
                skipped[col] = "column not found in X_train (dropped earlier?)"
                continue

            if semantic_type.get(col) not in TRANSFORMABLE_TYPES:
                skipped[col] = f"semantic type is '{semantic_type.get(col, 'unknown')}', only numeric_continuous / numeric_discrete are transformed"
                continue

            if not pd.api.types.is_numeric_dtype(X_train[col]) or pd.api.types.is_bool_dtype(X_train[col]):
                skipped[col] = f"dtype is {X_train[col].dtype}, not a numeric column"
                continue

            series = X_train[col].astype("float64")
            non_null = series.dropna()                 

            if non_null.nunique() < 2:
                skipped[col] = "constant (or all-missing) in train, nothing to transform"
                continue

            if method in NON_NEGATIVE_ONLY_METHODS and non_null.min() < 0:
                skipped[col] = f"{method} needs values >= 0, train min is {non_null.min():.4g}; use yeo_johnson instead"
                continue

           
            transformer = None
            if method == "log1p":
                transformed = np.log1p(series)
            elif method == "sqrt":
                transformed = np.sqrt(series)
            else:  # yeo_johnson
                transformer = PowerTransformer(method="yeo-johnson", standardize=False)  # scaling is a separate node
                transformed = pd.Series(
                    transformer.fit_transform(series.to_frame()).ravel(),
                    index=series.index,
                )

            if not np.isfinite(transformed.dropna()).all():
                skipped[col] = f"{method} produced inf/NaN values on train, column left unchanged"
                continue

            skew_before = float(non_null.skew())
            skew_after = float(transformed.dropna().skew())

            X_train[col] = transformed

            fitted_transforms[col] = {
                "method": method,
                "transformer": transformer,                       
                "lambda": float(transformer.lambdas_[0]) if transformer is not None else None,
            }
            report.append({
                "column": col,
                "method": method,
                "lambda": fitted_transforms[col]["lambda"],
                "skew_before_train": round(skew_before, 3),
                "skew_after_train": round(skew_after, 3),
            })

    return {
        "X_train": X_train,
        "numeric_transform_map": fitted_transforms,
        "transformed_numeric_report": report,
        "skipped_transform_numeric": skipped,
    }



