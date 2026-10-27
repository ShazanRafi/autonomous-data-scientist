import pandas as pd

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

def find_potential_ids(df):
    n_rows = len(df)

    if n_rows == 0:
        return []

    candidates = []

    for col in df.columns:
        series = df[col]
        name = str(col).lower()

        unique_ratio = series.nunique(dropna=True) / n_rows

        name_hint = any(
            word in name.split("_")
            for word in ["id", "uuid", "identifier", "key"]
        )

        if unique_ratio >= 0.98 or name_hint:
            candidates.append({
                "column": col,
                "unique_ratio": round(unique_ratio, 3),
                "name_hint": name_hint,
                "reason": (
                    "id_like_name_and_high_uniqueness"
                    if name_hint and unique_ratio >= 0.98
                    else "id_like_name"
                    if name_hint
                    else "high_uniqueness"
                )
            })

    return candidates


def find_text_columns(df, min_avg_len=30, min_avg_words=4):
    text_cols = []

    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].dropna().astype(str).str.strip()
        s = s[s.ne("")]

        if s.empty:
            continue

        avg_len = s.str.len().mean()
        avg_words = s.str.split().str.len().mean()

        if avg_len >= min_avg_len or avg_words >= min_avg_words:
            text_cols.append(col)

    return text_cols
