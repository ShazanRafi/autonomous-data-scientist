import re
import unicodedata
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import pint
ureg = pint.UnitRegistry()

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

def remove_rows_with_missing_target(X_train: pd.DataFrame, y_train: pd.Series):

    mask = y_train.notna()
    return X_train.loc[mask], y_train.loc[mask]

def remove_rows_that_are_completely_empty(X_train: pd.DataFrame, y_train: pd.Series):
    
    keep = X_train.notna().any(axis=1)
    return X_train.loc[keep], y_train.loc[keep]

def replace_infinite_values_with_nan(X_train: pd.DataFrame) -> pd.DataFrame:

    X_train = X_train.copy()
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns
    X_train[numeric_columns] = X_train[numeric_columns].replace([np.inf, -np.inf], np.nan)

    return X_train



def trim_whitespace(X_train: pd.DataFrame) -> pd.DataFrame:

    X_train = X_train.copy()
    for col in _text_columns(X_train):
        X_train[col] = _map_strings(X_train[col], str.strip)

    return X_train

def standardize_text_case(X_train: pd.DataFrame) -> pd.DataFrame:

    X_train = X_train.copy()
    for col in _text_columns(X_train):
        X_train[col] = _map_strings(X_train[col], str.lower)

    return X_train


_SPACE_LIKE = re.compile(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
_ZERO_WIDTH = re.compile(r"[\u200b-\u200d\u2060\ufeff]")

def _clean_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _SPACE_LIKE.sub(" ", text)      
    return _ZERO_WIDTH.sub("", text)       
 
def fix_unicode_and_encoding_issues(X_train: pd.DataFrame) -> pd.DataFrame:
    X_train = X_train.copy()
    for col in _text_columns(X_train):
        X_train[col] = _map_strings(X_train[col], _clean_unicode)
    return X_train

 

def remove_special_characters(X_train: pd.DataFrame) -> pd.DataFrame:
    
    X_train = X_train.copy()
    for col in _text_columns(X_train):
        X_train[col] = _map_strings(X_train[col], lambda s: re.sub(r"[^\w\s]", "", s))
    return X_train

def merge_spelling_variants_of_same_category(X_train: pd.DataFrame, threshold: int = 90, max_unique_values: int = 50) -> pd.DataFrame:
    
    X_train = X_train.copy()
 
    for col in _text_columns(X_train):
        counts = X_train[col].value_counts()        
        if len(counts) < 2 or len(counts) > max_unique_values:
            continue
 
        canonical_values: list = []
        mapping: dict = {}
        for value in counts.index:
            match = None
            if canonical_values:
                match = process.extractOne(
                    str(value), [str(c) for c in canonical_values],
                    scorer=fuzz.ratio, score_cutoff=threshold,
                )
            if match:
                mapping[value] = canonical_values[match[2]]
            else:
                canonical_values.append(value)
 
        if mapping:
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
 
        if pd.api.types.is_numeric_dtype(series):
            unique_values = series.dropna().unique()
            if len(unique_values) > 0 and set(unique_values).issubset({0, 1}):
                X_train[col] = series.astype("boolean")
            continue
 
        normalized = series.astype("string").str.strip().str.casefold()
        if normalized.dropna().nunique() == 2:            
            mapped = normalized.map(BOOLEAN_MAPPING)
            if mapped.notna().sum() == normalized.notna().sum():
                X_train[col] = mapped.astype("boolean")
 
    return X_train


_US_THOUSANDS = r"[1-9]\d{0,2}(?:,\d{3})+|[1-9]\d?(?:,\d{2})+,\d{3}"   
_EU_THOUSANDS = r"[1-9]\d{0,2}(?:\.\d{3})+"                         
US_NUMBER = rf"^(?=.*\d)(?:{_US_THOUSANDS}|\d+)?(?:\.\d+)?$"      
EU_NUMBER = rf"^(?=.*\d)(?:{_EU_THOUSANDS}|\d+)?(?:,\d+)?$"       

def _text_to_numbers(cleaned: pd.Series, body: pd.Series, style: str) -> pd.Series:
    
    if style == "european":
        fits = body.str.match(EU_NUMBER, na=False)
        normalized = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        fits = body.str.match(US_NUMBER, na=False)
        normalized = cleaned.str.replace(",", "", regex=False)
 
    numbers = pd.to_numeric(normalized.where(fits), errors="coerce")
    scientific = pd.to_numeric(cleaned.where(~fits), errors="coerce")
    if scientific.notna().any():
        numbers = numbers.astype("Float64").fillna(scientific.astype("Float64"))
    return numbers

def convert_text_to_numeric(X_train: pd.DataFrame, number_format: str = "auto") -> pd.DataFrame:
   
    X_train = X_train.copy()
 
    for col in _text_columns(X_train):
        original_non_null = X_train[col].notna().sum()
        if original_non_null == 0:
            continue
 
        cleaned = X_train[col].astype("string").str.strip().str.replace(r"[₹$€£\s'’]", "", regex=True)
 
        if cleaned.str.match(r"^[+-]?0\d", na=False).any():
            continue
 
        body = cleaned.str.replace(r"^[+-]", "", regex=True)
 
        style = number_format
        if style == "auto":
            fits_us = body.str.match(US_NUMBER, na=False)
            fits_eu = body.str.match(EU_NUMBER, na=False)
            us_evidence = (fits_us & ~fits_eu).sum()
            eu_evidence = (fits_eu & ~fits_us).sum()
            if us_evidence and eu_evidence:
                continue
            style = "european" if eu_evidence else "us"
 
        converted = _text_to_numbers(cleaned, body, style)
        if converted.notna().sum() / original_non_null >= 0.90:
            X_train[col] = converted
 
    return X_train


def convert_percentage_text_to_number(X_train: pd.DataFrame) -> pd.DataFrame:
    
    X_train = X_train.copy()
    percentage_pattern = r"\s*(?:%|percent(?:age)?|per\s+cent|pct)\s*$"
 
    for col in _text_columns(X_train):
        values = X_train[col].astype("string").str.strip()
        non_null = values.notna().sum()
        if non_null == 0:
            continue
 
        marked = values.str.contains(percentage_pattern, case=False, regex=True, na=False)
        if marked.sum() < 0.5 * non_null:
            continue
 
        numbers = pd.to_numeric(values.str.replace(percentage_pattern, "", regex=True, case=False).str.replace(",", "", regex=False).str.strip(),errors="coerce").astype("float64")
        X_train[col] = numbers.where(~marked.to_numpy(), numbers / 100)
 
    return X_train


def convert_text_to_datetime(X_train: pd.DataFrame, dayfirst: bool = False) -> pd.DataFrame:
    
    X_train = X_train.copy()
 
    for col in _text_columns(X_train):
        values = X_train[col].astype("string").str.strip()
        non_null = values.notna().sum()
        if non_null == 0:
            continue
 
        if values.str.contains(r"\d", na=False).sum() / non_null < 0.90:
            continue
 
        converted = pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=dayfirst)
        if converted.notna().sum() / non_null >= 0.90:
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
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce").astype("float64")
 
            elif semantic_type == "numeric_discrete":
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce").astype("Int64")
 
            elif semantic_type in {"categorical_nominal", "categorical_ordinal", "binary"}:
                X_train[col] = X_train[col].astype("category")
 
            elif semantic_type in {"text", "id"}:
                series = X_train[col]
                if pd.api.types.is_float_dtype(series) and (series.dropna() % 1 == 0).all():
                    series = series.astype("Int64")      # avoid ids like "123.0"
                X_train[col] = series.astype("string")
 
            elif semantic_type == "datetime":
                X_train[col] = pd.to_datetime(X_train[col], errors="coerce")
 
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
        number_of_parts = len(new_columns)
 
        if number_of_parts < 2 or any(name in X_train.columns for name in new_columns):
            continue
 
        parts = (
            X_train[col].astype("string")
            .str.split(separator, n=number_of_parts - 1, regex=False, expand=True)
            .reindex(columns=range(number_of_parts))
        )
        for i, new_col in enumerate(new_columns):
            X_train[new_col] = parts[i].astype("string").str.strip()
 
        X_train = X_train.drop(columns=[col])
 
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

def remove_duplicate_rows_by_key_columns(X_train: pd.DataFrame, key_columns: list[str],keep: str = "first"):


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