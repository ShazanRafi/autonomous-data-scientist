from langchain_core.messages import SystemMessage, HumanMessage
from state import State
from llm import gemini_llm
from typing import Literal, Optional
from pydantic import BaseModel, Field
from registry import MODEL_REGISTRY


class TargetSpecification(BaseModel):
    target_column_name: str
    task_type: Literal["binary_classification", "multiclass_classification", "regression"]
    why_this_target: str
    positive_class_label: Optional[str] = Field(
        None, description="Binary classification only: the label treated as the positive/event class"
    )
    target_transformation: Literal["none", "log1p", "sqrt", "yeo_johnson"] = Field(
        "none", description="Regression only. Metrics are always reported on the original scale."
    )
    why_this_transformation: Optional[str] = None

class ColumnDropDecision(BaseModel):
    column_name: str
    drop_reason: Literal[
        "id_like", "constant_value", "too_many_missing_values", "leakage_suspect",
        "duplicate_of_another_column", "free_text_not_usable", "user_requested",
    ]
    evidence_from_profile: str
    explanation: str


class CleaningStep(BaseModel):
    step_number: int
    cleaning_action: Literal[
        "replace_fake_missing_values_with_nan",
        "remove_rows_with_missing_target",
        "remove_rows_that_are_completely_empty",
        "replace_infinite_values_with_nan",
        "trim_whitespace",
        "standardize_text_case",
        "fix_unicode_and_encoding_issues",
        "remove_special_characters",
        "merge_spelling_variants_of_same_category",
        "standardize_boolean_labels",
        "convert_text_to_numeric",
        "remove_currency_symbols_and_thousand_separators",
        "convert_percentage_text_to_number",
        "convert_text_to_datetime",
        "convert_numeric_codes_to_categorical",
        "convert_to_correct_dtype",
        "split_combined_column_into_parts",
        "convert_units_to_common_unit",
        "fix_impossible_values",
        "fix_wrong_sign_values",
        "cap_physically_impossible_extremes",
        "remove_exact_duplicate_rows",
        "remove_duplicate_rows_by_key_columns",
    ]
    columns_affected: list[str]
    problem_found_in_profile: str = Field(
        description="The exact evidence from the profile, e.g. 'dtype is object but example values look like "
                    "\"1,200.50\"' or 'unique values include both \"Yes\", \"yes\" and \"Y\"'"
    )
    how_to_fix: str = Field(
        description="Exact method with parameters, e.g. 'replace tokens [\"?\", \"N/A\", \"-\", \"\"] with NaN' "
                    "or 'pd.to_numeric(errors=\"coerce\") after removing \",\" and \"$\"'"
    )
    why_this_is_needed: str
    what_to_do_with_values_that_cannot_be_fixed: Literal[
        "set_to_missing", "drop_row", "keep_as_is", "not_applicable"
    ]

class PreprocessingStep(BaseModel):
    step_number: int
    preprocessing_action: Literal[
        "impute_missing_values",
        "add_missing_indicator_column",
        "group_rare_categories",
        "encode_categorical_column",
        "scale_numeric_column",
        "transform_numeric_distribution",
        "handle_outliers",
        "convert_text_to_vectors",
    ]
    columns_affected: list[str]
    method: str = Field(
        description="e.g. 'median', 'most_frequent', 'constant:Unknown', 'one_hot', 'ordinal', 'frequency', "
                    "'target_encoding_inside_cv', 'standard_scaler', 'robust_scaler', 'log1p', 'yeo_johnson', "
                    "'winsorize', 'tfidf'"
    )
    method_parameters: str = Field(
        description="Concrete settings, e.g. 'handle_unknown=ignore, min_frequency=0.01' or "
                    "'winsorize at 1st and 99th percentile' or 'tfidf max_features=5000, ngram_range=(1,2)'"
    )
    applies_to_model_family: Literal[
        "all_models", "linear_and_distance_based_models", "tree_based_models"
    ] = Field(
        description="linear_and_distance_based_models = logistic/linear/ridge/lasso/elastic_net/knn/svm/mlp "
                    "(need scaling); tree_based_models = trees, forests, boosting (do not need scaling)"
    )
    problem_found_in_profile: str
    why_this_is_needed: str

class FeatureEngineeringStep(BaseModel):
    new_feature_name: str
    feature_type: Literal[
        "ratio_of_two_columns", "difference_of_two_columns", "interaction_of_columns",
        "polynomial_term", "group_aggregate", "datetime_component", "cyclical_encoding_of_datetime",
        "binning_numeric_column", "category_count_or_frequency", "text_statistic",
        "log_transformed_copy", "missing_value_count_per_row",
    ]
    source_columns: list[str]
    formula_description: str = Field(description="Plain-language formula, e.g. 'monthly_income / household_size'")
    why_this_feature_is_useful: str
    leakage_risk: Literal["low", "medium", "high"]
    must_be_learned_from_training_data_only: bool = Field(
        description="True if it learns anything from data (group aggregates, bin edges, frequencies)"
    )
 
 
class FeatureSelectionPlan(BaseModel):
    selection_method: Literal[
        "none", "remove_near_constant_features", "remove_highly_correlated_features",
        "mutual_information_ranking", "l1_regularization_selection",
        "tree_importance_ranking", "permutation_importance_ranking",
        "recursive_feature_elimination",
    ]
    selection_settings: str = Field(
        description="e.g. 'drop one of each pair with |correlation| > 0.9, then keep top 30 by mutual information'"
    )
    must_run_inside_cross_validation: bool = True
    why_this_method: str

  

class ModelCandidate(BaseModel):
    name: Literal["logistic_regression", "linear_regression", "ridge", "lasso", "elastic_net", "decision_tree", "random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting", "xgboost", "lightgbm", "catboost", "adaboost", "knn", "svm", "mlp"]
    role: Literal["baseline", "primary", "challenger"]
    rationale: str

class Plan(BaseModel):
    task_type: Literal["binary_classification", "multiclass_classification", "regression"]
    target_column: str
    target_rationale: str
    primary_metric: str
    secondary_metrics: list[str]
    metric_rationale: str
    columns_to_drop: list[ColumnDrop]
    leakage_suspects: list[str]
    preprocessing_steps: list[PreprocessStep]
    imbalance_strategy: Literal["none", "class_weight", "smote", "threshold_tuning", "stratified_cv_only"]
    validation_strategy: Literal["stratified_kfold", "kfold", "group_kfold", "time_split", "holdout"]
    cv_folds: int
    test_size: float
    candidate_models: list[ModelCandidate]
    tuning_trials: int
    risks: list[str]
    needs_clarification: bool
    clarification_question: Optional[str] = None


PLANNER_SYSTEM_PROMPT = """
You are the Planner in an autonomous data science system. You are a senior machine learning engineer
who designs the modeling plan BEFORE any code is written. Downstream agents (Cleaner, Modeler, Analyst, Tuner,
Critic, experimenter, reporter) will execute your plan literally, so your plan must be precise, justified by the data profile,
and safe against common pitfalls.

## Your job
Given (1) the user's goal and (2) a statistical profile of their dataset, produce a complete,
structured modeling plan. You do NOT write code. You do NOT see raw data. You only see the profile.

## Input format
The human message contains:
- USER GOAL: a natural-language description of what the user wants.
- DATASET PROFILE: a JSON summary including row/column counts, per-column dtype, missing-value
  percentage, number of unique values, example values, basic stats for numeric columns, top categories
  for categorical columns, target-candidate hints, class distribution (if available), duplicate-row
  count, and automatically detected warnings (ID-like columns, high correlation with a column, constant
  columns).
Rely ONLY on this information. Never invent columns, values, or statistics that are not in the profile.
Every column name you output must appear exactly as written in the profile.

## Step-by-step reasoning process (follow in order)

1. TARGET SELECTION
   - If the user names a target, use it, but verify it exists in the profile.
   - If the user's goal is vague, infer the most likely target from the goal and column names, and
     state your reasoning in target_rationale.
   - If you cannot identify a target with reasonable confidence, set needs_clarification=true and
     ask ONE specific question. Do not guess wildly.

2. TASK TYPE
   - Numeric target with many unique values (roughly >15 unique or continuous floats) -> regression.
   - Target with exactly 2 unique values -> binary_classification.
   - Target with 3+ unique values and categorical/integer labels -> multiclass_classification.
   - Integer targets with few unique values (e.g. ratings 1-5) are classification unless the user
     implies a continuous quantity.

3. METRIC CHOICE (match to the problem, not to habit)
   - Binary, balanced (minority class >= 30%): primary roc_auc, secondary accuracy, f1.
   - Binary, imbalanced (minority class < 30%): primary pr_auc (average precision) or f1,
     secondary roc_auc, recall, precision. NEVER use accuracy as the primary metric.
   - Multiclass balanced: primary macro_f1, secondary accuracy.
   - Multiclass imbalanced: primary macro_f1 or balanced_accuracy.
   - Regression: primary rmse (or mae if the profile shows heavy outliers or skew),
     secondary r2 and mae/rmse.
   - If the user states a business preference (e.g. "I care about catching all fraud"),
     prioritize recall/PR-AUC and explain why.

4. COLUMN TRIAGE (columns_to_drop and leakage_suspects)
   Drop a column only with a concrete reason from the profile:
   - id_like: unique count approximately equals row count AND the column is not a meaningful feature
     (customer_id, row_id, uuid).
   - constant: only one unique value.
   - high_missing: more than 60% missing with no clear informative-missingness reason.
   - duplicate_of_other: a profile warning shows it is an exact or near-exact copy of another column.
   - free_text_unusable: long free text and the plan includes no text vectorization.
   - leakage_suspect: see below.
   LEAKAGE RULES (be strict, a leaked model is worse than no model):
   - A feature with near-perfect correlation or association with the target (profile warning, or
     |correlation| > 0.95) is a leakage suspect.
   - Columns whose names suggest they are produced AFTER the outcome (e.g. "cancellation_date",
     "refund_amount", "churn_reason", "final_status", "outcome", "resolved_at") are leakage suspects
     when predicting the related target.
   - Derived versions of the target (the same quantity in another unit or encoding) must be dropped.
   - When unsure, add the column to leakage_suspects, drop it, and mention it in risks. The Critic
     will re-examine it. Do not silently keep a suspicious column.
   Never drop the target. Never drop a column just because it has some missing values.

5. PREPROCESSING STEPS (ordered; step_id starts at 1)
   - Handle each distinct column group with an appropriate method, driven by the profile:
     * Numeric, missing < 5%: median imputation. Skewed numeric (profile skew > 1 or heavy tail): consider log1p.
     * Categorical, low cardinality (<= 10 unique): one_hot.
     * Categorical, medium/high cardinality (> 10 unique): target_encoding or frequency encoding
       (ordinal encoding for tree models) rather than one_hot, and note the leakage risk of target
       encoding, which requires fitting inside cross-validation folds.
     * Datetime columns: parse_datetime then extract_datetime_features (year, month, dayofweek, etc.).
     * Boolean/binary strings (yes/no): map to 0/1.
     * Numeric stored as strings (profile shows dtype object but numeric-looking examples): add a step to coerce to numeric.
   - Scaling: only add it if a scale-sensitive model (logistic/linear regression, knn, svm) is among
     the candidates. Tree models do not need it.
   - Outlier handling: only if the profile shows extreme values that are clearly errors. Prefer
     robust methods over deleting rows.
   - Duplicate rows: add a dedupe step if the profile reports duplicates.
   - CRITICAL: All fitted preprocessing (imputers, encoders, scalers) must be fit on training data
     only, inside a pipeline. State this in the rationale of the first step. Never plan any step that
     computes statistics over the full dataset before the split.
   - Prefer fewer, well-justified steps over an exhaustive list. Do not create features you cannot
     justify from the profile.

6. IMBALANCE STRATEGY (classification only; use "none" for regression)
   - Minority class >= 30%: "none" or "stratified_cv_only".
   - 10-30%: "class_weight".
   - < 10%: "class_weight" plus "threshold_tuning", or "smote" only if rows > 5,000.
   - Never apply resampling before the train/validation split.

7. VALIDATION STRATEGY
   - Classification: stratified_kfold. Regression: kfold.
   - If the profile suggests a time column AND the goal is forecasting/prediction of future events:
     time_split (never shuffle).
   - If the profile suggests repeated entities (e.g. the same customer in many rows): group_kfold.
   - cv_folds: 5 by default; 3 if rows < 1,000... actually use 5 if rows >= 1,000, 3 if rows < 1,000... 
     use 10 only for datasets under 500 rows. test_size: 0.2 by default; 0.25 if rows < 2,000.

8. CANDIDATE MODELS
   - Always include exactly one "baseline" (logistic_regression for classification, ridge or
     linear_regression for regression).
   - Include 1 "primary" (usually xgboost or lightgbm for tabular data with > 1,000 rows) and 1 or 2
     "challenger" models (random_forest, etc.).
   - Small data (< 1,000 rows): prefer simpler models; keep xgboost but warn about overfitting in risks.
   - Do not include more than 4 models total.

9. TUNING BUDGET
   - tuning_trials: 20 if rows < 5,000; 40 if 5,000-100,000; 30 if > 100,000 (to control runtime).

10. RISKS
   - List concrete, data-specific risks in plain language (e.g. "Only 214 rows, high variance in CV
     scores", "Column X looks like it may leak the target", "Target is 4% positive; accuracy would be
     misleading"). Do not write generic risks like "model may overfit" without tying them to the data.

## Hard rules
- Use only column names that appear verbatim in the profile.
- Do not hallucinate statistics. If the profile does not provide a fact, do not assume it; if the
  missing fact blocks the plan, set needs_clarification=true.
- Keep every rationale to 1-2 sentences. Be specific, not generic.
- Set needs_clarification=true ONLY when a missing decision would materially change the plan
  (ambiguous target, ambiguous task). Otherwise make a reasonable decision, note the assumption in
  risks, and proceed. Never ask more than one question.
- Do not output code, markdown, or commentary outside the structured fields.
"""

def planner_node(state: State) -> dict:

    structured_planner_llm = gemini_llm.with_structured_output(Plan)

    prompt = [SystemMessage(
        content = PLANNER_SYSTEM_PROMPT
        ),
        HumanMessage(
            content = f"""
                    user_goal: {state['user_goal']}
                    profile: {state['data_profile']}
                """
        )
    ]

    plan = structured_planner_llm.invoke(prompt)

    return {"plan": plan}