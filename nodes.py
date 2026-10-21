from state import State
from langchain_community.document_loaders import CSVLoader

def loader_node(state: State) -> dict:
    filepath = state[filepath]
    loader = CSVLoader(file_path=filepath)
    docs = loader.load()
    return {'filedata': docs}

def profiler_node(state: State) -> dict:
    pass