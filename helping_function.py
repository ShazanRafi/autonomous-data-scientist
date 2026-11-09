import unicodedata
import pandas as pd
from rapidfuzz import process, fuzz

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


def replace_fake_missing_values_with_nan(dataframe: pd.DataFrame, fake_missing_values: set[str] | None = None, numeric_sentinels: list[float] | None = None) -> pd.DataFrame:

    DEFAULT_FAKE_MISSING = {
        "", " ", "?", "-", "--", "---", ".", "..", "n/a", "na", "n.a.", "nan",
        "null", "none", "nil", "missing", "unknown", "undefined", "not available",
        "#n/a", "#na", "#null!", "(blank)", "blank", "tbd",
        "not applicable", "not provided", "not specified", "not recorded",
        "not collected", "not reported", "not disclosed", "not documented"
    }

    if fake_missing_values:
        DEFAULT_FAKE_MISSING.update(fake_missing_values)
    if numeric_sentinels:
        DEFAULT_FAKE_MISSING.update(map(str, numeric_sentinels))

    for value in DEFAULT_FAKE_MISSING:
        dataframe.replace(value, pd.NA, inplace=True)
        return dataframe

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
    X_train.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
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
            .str.replace("%", "", regex=False)
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




    




