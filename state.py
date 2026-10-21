from typing import TypedDict, List
from langchain_core.documents import Document

class State(TypedDict):
    filepath: str
    filedata: List[Document]
