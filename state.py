from typing import TypedDict, List
import pandas as pd
from langchain_core.documents import Document

class State(TypedDict):
    filepath: str
    dataframe: pd.DataFrame
    data_profile: dict
