import unicodedata
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import pint
ureg = pint.UnitRegistry()

def choose_outlier_method(series):
    skewness = series.skew()

    if abs(skewness) <= 0.5:
        return "z_score"
    else:
        return "iqr"

def detect_outliers(series):
    method = choose_outlier_method(series)

    if method == "z_score":
        mean = series.mean()
        std = series.std()

        if std == 0 or pd.isna(std):
            return series != series  # all False

        return abs((series - mean) / std) > 3

    else:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return (series < lower) | (series > upper)

def percentage_of_outliers(series):
    series = series.dropna()

    if len(series) == 0:
        return 0.0

    outliers = detect_outliers(series)
    return round(outliers.sum() / len(series) * 100, 2)

#********************************************************************DATA CLEANING HELPERS********************************************************************

TEXT_DTYPES = ["object", "string"]
 
def _text_columns(df: pd.DataFrame):
    return df.select_dtypes(include=TEXT_DTYPES).columns
 
def _map_strings(series: pd.Series, func) -> pd.Series:
    return series.map(lambda v: func(v) if isinstance(v, str) else v)


DEFAULT_FAKE_MISSING = frozenset({
        "", " ", "?", "-", "--", "---", ".", "..", "n/a", "na", "n.a.", "nan",
        "null", "none", "nil", "missing", "unknown", "undefined", "not available",
        "#n/a", "#na", "#null!", "(blank)", "blank", "tbd",
        "not applicable", "not provided", "not specified", "not recorded",
        "not collected", "not reported", "not disclosed", "not documented"
    })

def replace_fake_missing_values_with_nan(X_train: pd.DataFrame, fake_missing_values: set[str] | None = None, numeric_sentinels: list[float] | None = None, sentinel_columns: list[str] | None = None) -> pd.DataFrame:

    X_train = X_train.copy()
 
    tokens = set(DEFAULT_FAKE_MISSING)

    if fake_missing_values:
        tokens |= {str(tkn).strip().lower() for tkn in fake_missing_values}
 
    for col in _text_columns(X_train):
        normalized = X_train[col].astype("string").str.strip().str.lower()
        X_train[col] = X_train[col].mask(normalized.isin(tokens).fillna(False), np.nan)
 
    if numeric_sentinels:
        columns = sentinel_columns if sentinel_columns else X_train.columns
        for col in columns:
            if col in X_train.columns and pd.api.types.is_numeric_dtype(X_train[col]):
                X_train[col] = X_train[col].mask(X_train[col].isin(numeric_sentinels))
 
    return X_train

def remove_rows_with_missing_target(X_train: pd.DataFrame, y_train: pd.DataFrame):
    mask = y_train.notna()

    X_train = X_train.loc[mask]
    y_train = y_train.loc[mask]

    return X_train, y_train

def remove_rows_that_are_completely_empty(X_train: pd.DataFrame, y_train: pd.DataFrame):
    train_df = pd.concat([X_train, y_train], axis=1)

    train_df = train_df.dropna(how="all")
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    return X_train, y_train

def replace_infinite_values_with_nan(X_train: pd.DataFrame) -> pd.DataFrame:
    X_train = X_train.copy()
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns
    X_train[numeric_columns] = X_train[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return X_train

def trim_whitespace_from_string_columns(X_train: pd.DataFrame) -> pd.DataFrame:
    string_columns = X_train.select_dtypes(include="object").columns
    X_train[string_columns] = X_train[string_columns].apply(lambda x: x.str.strip())
    return X_train

def standardize_text_case(X_train: pd.DataFrame) -> pd.DataFrame:
    string_columns = X_train.select_dtypes(include="object").columns
    X_train[string_columns] = X_train[string_columns].apply(lambda x: x.str.lower())

    return X_train



def fix_unicode_and_encoding_issues(X_train: pd.DataFrame) -> pd.DataFrame:

    string_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in string_columns:
        X_train[col] = X_train[col].map(
            lambda x: unicodedata.normalize("NFC", x)
            if isinstance(x, str) else x
        )

    return X_train

def remove_special_characters(X_train: pd.DataFrame) -> pd.DataFrame:
    string_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in string_columns:
        X_train[col] = X_train[col].str.replace(r"[^\w\s]", "", regex=True)

    return X_train

def merge_spelling_variants_of_same_category(X_train: pd.DataFrame, threshold: int = 90) -> pd.DataFrame:

    categorical_columns = X_train.select_dtypes(include=["object", "category", "string"]).columns

    for col in categorical_columns:
        values = X_train[col].dropna().unique().tolist()
        canonical_values = []
        mapping = {}
        for value in values:
            match = process.extractOne(
                str(value),
                canonical_values,
                scorer=fuzz.ratio,
                score_cutoff=threshold
            )
            if match:
                mapping[value] = match[0]
            else:
                canonical_values.append(value)
                mapping[value] = value

        X_train[col] = X_train[col].replace(mapping)

    return X_train


BOOLEAN_MAPPING = {
    "yes": True, "y": True, "true": True, "t": True,
    "1": True, "on": True, "enabled": True, "active": True,
    "no": False, "n": False, "false": False, "f": False,
    "0": False, "off": False, "disabled": False, "inactive": False,
}

def standardize_boolean_labels(X_train: pd.DataFrame) -> pd.DataFrame:
    X_train = X_train.copy()

    for col in X_train.columns:
        series = X_train[col]

        if pd.api.types.is_bool_dtype(series):
            continue

        unique_values = series.dropna().unique()

        if pd.api.types.is_numeric_dtype(series):
            if len(unique_values) > 0 and set(unique_values).issubset({0, 1}):
                X_train[col] = series.astype("boolean")
            continue

        if len(unique_values) == 2:
            normalized = series.astype("string").str.strip().str.casefold()
            mapped = normalized.map(BOOLEAN_MAPPING)

            if mapped.notna().sum() == normalized.notna().sum():
                X_train[col] = mapped.astype("boolean")

    return X_train

import pandas as pd

def convert_text_to_numeric(X_train: pd.DataFrame) -> pd.DataFrame:
    X_train = X_train.copy()

    for col in X_train.select_dtypes(include=["object", "string"]).columns:

        cleaned = (
            X_train[col]
            .astype("string")
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace(r"[₹$€£]", "", regex=True)
        )

        converted = pd.to_numeric(cleaned, errors="coerce")

        original_non_null = X_train[col].notna().sum()

        if original_non_null == 0:
            continue

        conversion_ratio = converted.notna().sum() / original_non_null

        if conversion_ratio >= 0.90:
            X_train[col] = converted

    return X_train


def convert_percentage_text_to_number(X_train: pd.DataFrame) -> pd.DataFrame:
    X_train = X_train.copy()

    percentage_pattern = r"\s*(?:%|percent(?:age)?|per\s+cent|pct)\s*$"

    for col in X_train.select_dtypes(include=["object", "string"]).columns:

        values = X_train[col].astype("string").str.strip()

        percentage_mask = values.str.contains(
            percentage_pattern,
            case=False,
            regex=True,
            na=False
        )

        if not percentage_mask.any():
            continue

        numeric_values = pd.to_numeric(
            values.str.replace(
                percentage_pattern, "", regex=True, case=False
            ).str.strip(),
            errors="coerce"
        )

        X_train.loc[percentage_mask, col] = (
            numeric_values[percentage_mask] / 100
        )

    return X_train

def convert_text_to_datetime(X_train: pd.DataFrame) -> pd.DataFrame:
    X_train = X_train.copy()

    for col in X_train.select_dtypes(include=["object", "string"]).columns:

        values = X_train[col].astype("string").str.strip()

        converted = pd.to_datetime(
            values,
            errors="coerce",
            format="mixed"
        )

        original_non_null = values.notna().sum()

        if original_non_null == 0:
            continue

        conversion_ratio = converted.notna().sum() / original_non_null

        if conversion_ratio >= 0.90:
            X_train[col] = converted

    return X_train

def convert_numeric_codes_to_categorical(X_train: pd.DataFrame, column_types: dict) -> pd.DataFrame:

    X_train = X_train.copy()

    for col, semantic_type in column_types.items():

        if col not in X_train.columns:
            continue

        if semantic_type in {"categorical_nominal", "categorical_ordinal", "binary"}:

            if pd.api.types.is_numeric_dtype(X_train[col]):
                X_train[col] = X_train[col].astype("category")

    return X_train

def convert_to_correct_dtype(X_train: pd.DataFrame, column_types: dict) -> pd.DataFrame:

    X_train = X_train.copy()

    for col, semantic_type in column_types.items():

        if col not in X_train.columns:
            continue

        try:
            if semantic_type == "numeric_continuous":
                X_train[col] = pd.to_numeric(
                    X_train[col], errors="coerce"
                ).astype("float64")

            elif semantic_type == "numeric_discrete":
                X_train[col] = pd.to_numeric(
                    X_train[col], errors="coerce"
                ).astype("Int64")

            elif semantic_type in {
                "categorical_nominal",
                "categorical_ordinal",
                "binary"
            }:
                X_train[col] = X_train[col].astype("category")

            elif semantic_type in {"text", "id"}:
                X_train[col] = X_train[col].astype("string")

            elif semantic_type == "datetime":
                X_train[col] = pd.to_datetime(
                    X_train[col], errors="coerce"
                )

            elif semantic_type == "constant":
                continue

        except (ValueError, TypeError, OverflowError):
            continue

    return X_train

def split_combined_column_into_parts(X_train: pd.DataFrame, split_config: dict) -> pd.DataFrame:

    X_train = X_train.copy()

    for col, config in split_config.items():

        if col not in X_train.columns:
            continue

        separator = config["separator"]
        new_columns = config["new_columns"]

        try:
            split_values = (
                X_train[col]
                .astype("string")
                .str.split(separator, regex=False, expand=True)
            )

            if split_values.shape[1] != len(new_columns):
                continue

            for i, new_col in enumerate(new_columns):
                if new_col in X_train.columns:
                    continue

                X_train[new_col] = split_values[i].str.strip()

            X_train.drop(columns=[col], inplace=True)

        except (ValueError, TypeError, KeyError):
            continue

    return X_train

def convert_units_to_common_unit(X_train: pd.DataFrame, unit_config: dict) -> pd.DataFrame:

    X_train = X_train.copy()

    for col, config in unit_config.items():

        if col not in X_train.columns:
            continue

        target_unit = config["target_unit"]
        source_unit = config.get("source_unit")

        converted_values = []
        valid = True

        for value in X_train[col]:

            if pd.isna(value):
                converted_values.append(pd.NA)
                continue

            try:
                if isinstance(value, (int, float)):
                    if source_unit is None:
                        valid = False
                        break

                    quantity = ureg.Quantity(value, source_unit)

                else:
                    quantity = ureg.Quantity(str(value).strip())

                converted = quantity.to(target_unit)
                converted_values.append(float(converted.magnitude))

            except (ValueError, TypeError, pint.errors.PintError):
                valid = False
                break

        if valid:
            X_train[col] = converted_values

    return X_train

import pandas as pd

def fix_impossible_values(X_train: pd.DataFrame, constraint_config: dict) -> pd.DataFrame:

    X_train = X_train.copy()

    for col, constraints in constraint_config.items():

        if col not in X_train.columns:
            continue

        invalid_mask = pd.Series(False, index=X_train.index)

        min_value = constraints.get("min")
        max_value = constraints.get("max")

        if min_value is not None:
            invalid_mask |= X_train[col] < min_value

        if max_value is not None:
            invalid_mask |= X_train[col] > max_value

        X_train.loc[invalid_mask, col] = pd.NA

    return X_train

import pandas as pd

def fix_wrong_sign_values(X_train: pd.DataFrame,non_negative_columns: list) -> pd.DataFrame:

    X_train = X_train.copy()

    for col in non_negative_columns:

        if col not in X_train.columns:
            continue

        invalid_mask = X_train[col] < 0

        X_train.loc[invalid_mask, col] = pd.NA

    return X_train

import pandas as pd

def cap_physically_impossible_extremes(X_train, limits_config):
    X_train = X_train.copy()

    for col, limits in limits_config.items():
        if col not in X_train.columns:
            continue

        min_value = limits.get("min")
        max_value = limits.get("max")

        invalid_mask = pd.Series(False, index=X_train.index)

        if min_value is not None:
            invalid_mask |= X_train[col] < min_value

        if max_value is not None:
            invalid_mask |= X_train[col] > max_value

        X_train.loc[invalid_mask, col] = pd.NA

    return X_train

def remove_exact_duplicate_rows(X_train: pd.DataFrame) -> pd.DataFrame:
    return X_train.drop_duplicates(keep="first").reset_index(drop=True)

def remove_duplicate_rows_by_key_columns(X_train: pd.DataFrame,key_columns: list[str],keep: str = "first"):


    missing_columns = set(key_columns) - set(X_train.columns)

    if missing_columns:
        raise ValueError(
            f"Key columns not found: {missing_columns}"
        )

    if not key_columns:
        raise ValueError("At least one key column is required.")

    before = len(X_train)

    X_train = X_train.drop_duplicates(
        subset=key_columns,
        keep=keep,
        ignore_index=True
    )

    removed = before - len(X_train)

    return X_train, {
        "rows_before": before,
        "rows_after": len(X_train),
        "duplicates_removed": removed,
        "duplicate_percentage": round(
            removed / before * 100, 2
        ) if before else 0
    }








    




