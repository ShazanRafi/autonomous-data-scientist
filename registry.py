from enum import Enum
from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge,
                                  RidgeClassifier, Lasso, ElasticNet)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              AdaBoostClassifier, AdaBoostRegressor)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

MODEL_REGISTRY = {

    "logistic_regression": dict(family="linear", estimators={"clf": LogisticRegression},
        needs_scaling=True,  native_missing=False, native_categorical=False, max_rows=None,
        notes="Fast, interpretable baseline for classification."),
    "linear_regression":   dict(family="linear", estimators={"reg": LinearRegression},
        needs_scaling=False, native_missing=False, native_categorical=False, max_rows=None,
        notes="OLS baseline. Sensitive to multicollinearity and outliers."),
    "ridge":               dict(family="linear", estimators={"reg": Ridge, "clf": RidgeClassifier},
        needs_scaling=True,  native_missing=False, native_categorical=False, max_rows=None,
        notes="L2-regularized linear model. Good when features are correlated."),
    "lasso":               dict(family="linear", estimators={"reg": Lasso},
        needs_scaling=True,  native_missing=False, native_categorical=False, max_rows=None,
        notes="L1-regularized. Does feature selection on wide data."),
    "elastic_net":         dict(family="linear", estimators={"reg": ElasticNet},
        needs_scaling=True,  native_missing=False, native_categorical=False, max_rows=None,
        notes="Mix of L1 and L2. Good for wide, correlated features."),


    "decision_tree":       dict(family="tree", estimators={"clf": DecisionTreeClassifier, "reg": DecisionTreeRegressor},
        needs_scaling=False, native_missing=False, native_categorical=False, max_rows=None,
        notes="Interpretable but high variance. Useful as a simple baseline."),
    "random_forest":       dict(family="bagging", estimators={"clf": RandomForestClassifier, "reg": RandomForestRegressor},
        needs_scaling=False, native_missing=False, native_categorical=False, max_rows=None,
        notes="Robust, low-tuning strong default."),
    "extra_trees":         dict(family="bagging", estimators={"clf": ExtraTreesClassifier, "reg": ExtraTreesRegressor},
        needs_scaling=False, native_missing=False, native_categorical=False, max_rows=None,
        notes="Faster, more random than random forest. Good on noisy data."),

  
    "gradient_boosting":   dict(family="boosting", estimators={"clf": GradientBoostingClassifier, "reg": GradientBoostingRegressor},
        needs_scaling=False, native_missing=False, native_categorical=False, max_rows=50_000,
        notes="Classic sklearn GBM. Slow on large data."),
    "hist_gradient_boosting": dict(family="boosting", estimators={"clf": HistGradientBoostingClassifier, "reg": HistGradientBoostingRegressor},
        needs_scaling=False, native_missing=True,  native_categorical=False, max_rows=None,
        notes="Fast sklearn boosting. Handles missing values natively."),
    "xgboost":             dict(family="boosting", estimators={"clf": XGBClassifier, "reg": XGBRegressor},
        needs_scaling=False, native_missing=True,  native_categorical=False, max_rows=None,
        notes="Strong default for tabular data."),
    "lightgbm":            dict(family="boosting", estimators={"clf": LGBMClassifier, "reg": LGBMRegressor},
        needs_scaling=False, native_missing=True,  native_categorical=True,  max_rows=None,
        notes="Very fast on large data. Can overfit on tiny datasets."),
    "catboost":            dict(family="boosting", estimators={"clf": CatBoostClassifier, "reg": CatBoostRegressor},
        needs_scaling=False, native_missing=True,  native_categorical=True,  max_rows=None,
        notes="Best with many high-cardinality categorical columns."),
    "adaboost":            dict(family="boosting", estimators={"clf": AdaBoostClassifier, "reg": AdaBoostRegressor},
        needs_scaling=False, native_missing=False, native_categorical=False, max_rows=None,
        notes="Older boosting method. Mostly useful as a challenger."),


    "knn":                 dict(family="distance", estimators={"clf": KNeighborsClassifier, "reg": KNeighborsRegressor},
        needs_scaling=True,  native_missing=False, native_categorical=False, max_rows=50_000,
        notes="Slow at prediction on large data. Needs scaling."),
    "svm":                 dict(family="kernel", estimators={"clf": SVC, "reg": SVR},
        needs_scaling=True,  native_missing=False, native_categorical=False, max_rows=20_000,
        notes="Strong on small, clean data. Does not scale to large datasets."),
    "mlp":                 dict(family="neural", estimators={"clf": MLPClassifier, "reg": MLPRegressor},
        needs_scaling=True,  native_missing=False, native_categorical=False, max_rows=None,
        notes="Basic neural network. Rarely beats boosting on tabular data."),
}