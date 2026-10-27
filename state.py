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
    
