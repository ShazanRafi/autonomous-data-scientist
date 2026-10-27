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