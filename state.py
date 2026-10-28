from typing import TypedDict, List
import pandas as pd
from langchain_core.documents import Document
from agent.planner import Plan

class State(TypedDict):
    filepath: str
    dataframe: pd.DataFrame
    data_profile: dict
    user_goal: str
    plan: Plan
    selected_model: list[str]
    should_scale: bool
    scaling_report: dict
    target_column = str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    
