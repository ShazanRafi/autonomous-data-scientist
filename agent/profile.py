from state import State
import pandas as pd
from helping_function import detect_outliers, find_potential_ids, find_text_columns, percentage_of_outliers
from pandas.api.types import is_datetime64_any_dtype
from pydantic import BaseModel
from typing import Literal, Optional
from llm import gemini_llm
from langchain_core.messages import SystemMessage, HumanMessage
import json


def loader_node(state: State) -> dict:
    filepath = state[filepath]
    df = pd.read_csv(filepath)

    return{'dataframe': df}



def column_evidence(df, col):
    column = df[col]
    num_rows = len(column)

    llm_registry_profile = {
        "name": col,
        "dtype": str(column.dtype),
        "null_percentage": round(column.isna().mean() * 100, 2),
        "n_unique": int(column.nunique()),
        "unique_ratio": round(column.nunique() / num_rows, 4),
        "sample_values": column.dropna().astype(str).unique()[:8].tolist(),
        "top_values": column.value_counts().head(5).to_dict() if column.nunique() < 50 else None,
        "avg_str_len": round(column.dropna().astype(str).str.len().mean(), 1) if column.dtype == "object" else None,
    }

    return {'llm_registry_profile': llm_registry_profile}



class ColumnInfo(BaseModel):
    name: str
    semantic_type: Literal["numeric_continuous", "numeric_discrete", "categorical_nominal",
                           "categorical_ordinal", "binary", "text", "datetime", "id", "constant"]
    role: Literal["feature", "target_candidate", "drop"]
    ordinal_order: Optional[list[str]] = None  
    suggested_encoding: Literal["onehot", "ordinal", "target", "frequency", "tfidf", "none", "drop"]
    confidence: float
    reason: str

class ColumnRegistry(BaseModel):
    columns: list[ColumnInfo]

COLUMN_CATEGORICAL_PROMPT = """
You are a senior data scientist whose only job is to classify the columns of a tabular dataset so that downstream preprocessing can be done by deterministic code. You do not clean, transform, or model the data. You only label columns.

# Input
You will receive:
1. DATASET_CONTEXT: row count, column count, and optionally the target column name and problem type.
2. COLUMN_EVIDENCE: one block per column with: name, dtype, null_pct, n_unique, unique_ratio, sample_values, top_values (value counts, when cardinality is low), and avg_str_len (for string columns).

You never see the full dataset. Base every decision only on the evidence provided. Do not invent values, categories, or statistics that are not in the evidence.

# Output
Return one entry for every column, exactly once, in the original order. Each entry has:
- name: exact column name as given. Never rename, merge, or skip.
- semantic_type: one of numeric_continuous, numeric_discrete, categorical_nominal, categorical_ordinal, binary, text, datetime, id, constant
- role: one of feature, target_candidate, drop
- ordinal_order: list of values from lowest to highest. Only for categorical_ordinal, otherwise null. Every value must appear in the column's evidence. Include every distinct value you can see; if you cannot see all values (n_unique larger than the values shown) do not classify as ordinal.
- suggested_encoding: one of onehot, ordinal, target, frequency, tfidf, none, drop
- confidence: 0.0 to 1.0
- reason: one short sentence citing the specific evidence (name, values, ratio) behind the decision.

# Classification rules
Judge by meaning first (column name + sample values), then by statistics. dtype alone is never enough: a numeric dtype can be an ID, a category code, or a binary flag; an object dtype can be a number, a date, or free text.

- id: unique_ratio near 1 AND the name or values look like an identifier (id, uuid, code, index, number), or a numeric sequence with no analytical meaning. Role = drop, encoding = drop.
- constant: n_unique <= 1. Role = drop, encoding = drop.
- binary: exactly 2 distinct non-null values (yes/no, 0/1, true/false, male/female). Encoding = ordinal (map to 0/1) or none if already 0/1.
- categorical_nominal: a limited set of labels with no natural order (city, color, brand, gender). 
- categorical_ordinal: labels with a clear natural order (low/medium/high, grade levels, education level, satisfaction scales, size S/M/L/XL). Provide ordinal_order. If the order is not obvious from the values, classify as nominal instead of guessing.
- numeric_discrete: integer counts or small-range integers where magnitude matters (number of rooms, age in years, tenure in months). If integer codes represent labels (e.g. 1=red, 2=blue, zip codes, category codes), classify as categorical, not numeric.
- numeric_continuous: measurements or amounts with many distinct values (price, weight, temperature).
- datetime: values that represent dates or timestamps, even when stored as strings. Encoding = none (feature extraction is handled later).
- text: free-form strings where values are mostly unique and long (reviews, descriptions, comments, addresses). Typically high unique_ratio AND high avg_str_len. Short high-cardinality strings (names, cities) are not text.

# Encoding rules
- numeric_continuous, numeric_discrete, datetime: none
- binary: ordinal or none
- categorical_nominal: onehot if n_unique <= 15; frequency if 16 to 50; target if more than 50 (target only when a target column is provided, otherwise frequency)
- categorical_ordinal: ordinal
- text: tfidf
- id, constant: drop

# Role rules
- feature: usable as a model input.
- target_candidate: use only if no target column was provided and the column plausibly is the outcome being predicted (e.g. churn, price, survived, label, status). If a target column is provided, mark exactly that column as target_candidate and no others.
- drop: ids, constants, and columns with null_pct above 60 unless they carry clear meaning. 

# Confidence
- 0.9 to 1.0: name and values clearly agree.
- 0.7 to 0.89: probable, minor ambiguity.
- Below 0.7: genuinely ambiguous (e.g. an integer column that could be a count or a code, a categorical whose order is uncertain). Do not inflate confidence. Low confidence will trigger human review, which is the correct outcome for ambiguous columns.

# Hard constraints
- One entry per column, exact names, no extras, no omissions.
- Never output an ordinal_order containing values not present in the evidence.
- Do not recommend dropping a column merely because it has missing values; missing-value handling is a later step.
- Do not leak the target: columns that look derived from the target (e.g. "churn_date" when target is "churned") should be flagged in the reason and set to role = drop.
- Keep each reason to one sentence.
"""

def column_categorizing_node(state:State)->dict:
    df = state["dataframe"] 
    evidence_list = [column_evidence(df, col) for col in df.columns]   


    prompt = [SystemMessage(
            content = COLUMN_CATEGORICAL_PROMPT
            ),
            HumanMessage(
                content = f"""
                        DATASET_CONTEXT:
                            rows: {df.shape[0]}
                            columns: {df.shape[1]}
                            target_column: {state.get("target_column") or "not provided"}
                        
                        COLUMN_EVIDENCE: {json.dumps(evidence_list, indent=2, default=str)}
                    """
            )
        ]
    
    structured_planner_llm = gemini_llm.with_structured_output(ColumnRegistry)
    result = structured_planner_llm.invoke(prompt)
    registry = {column.name: column.model_dump() for column in result.columns}

    return {"column_registry": registry}



def profiler_node(state: State) -> dict:
    df = state['dataframe']
    no_of_rows, no_of_columns = df.shape
    columns = df.columns.tolist()
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

    outlier_percentage = {}
    for col in df.select_dtypes(include="number").columns:
        outlier[col] = percentage_of_outliers(df[col])

    column_dtypes = {}
    column_roles = {}
    categorical_columns = []
    column_registry = state.get("column_registry")
    for col, info in column_registry.items():
        column_dtypes[col] = info["semantic_type"]
        column_roles[col] = info["role"]
        if info["semantic_type"].startswith("categorical"):
            categorical_columns.append(col)

    categorical_value_counts = {}
    for col in categorical_columns:
        categorical_value_counts[col] = df[col].value_counts(dropna=False).head(30).to_dict()


    sample_values = {}
    for col in df:
        sample_values[col] = df[col].dropna().astype(str).unique()[:8].tolist()

    empty_rows_and_columns = {
        'empty_rows': int(df.isna().all(axis=1).sum()),
        'empty_columns': [col for col in df.columns if df[col].isna().all()]
    }


    profile = {
        'no_of_rows': no_of_rows,
        'no_of_columns': no_of_columns,
        'all_columns': columns,
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
        'Outlier': outlier,
        'percentage_of_outlier': outlier_percentage,
        'column_dtypes': column_dtypes,
        'column_roles': column_roles,
        'sample_values_per_column': sample_values,
        'categorical_value_counts': categorical_value_counts,
        'empty_rows_and_columns': empty_rows_and_columns
    }

    return{'data_profile': profile}

