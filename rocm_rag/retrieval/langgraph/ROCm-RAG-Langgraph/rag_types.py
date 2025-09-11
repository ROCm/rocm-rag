from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    urls: List[str]
    answer: Optional[str]
    history: List[dict]
    stream: bool