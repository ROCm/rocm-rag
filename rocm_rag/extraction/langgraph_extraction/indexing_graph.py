from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from rocm_rag.utils.langgraph_vectorstore import RAGVectorStore
from rocm_rag import config
from weaviate.classes.query import Filter
import torch
import nltk
nltk.download("punkt")

# in this implementation, we use the SemanticChunker from langchain_experimental
# new embedding is not used for merging chunks
# so we need to regenerate embeddings for the merged chunks
class RAGSemanticChunker:
    def __init__(
        self,
        embedder: OpenAIEmbeddings,
        buffer_size: int = 1,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = None,
    ):
        
        self.chunker = SemanticChunker(
            embeddings=embedder,
            buffer_size=buffer_size,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

    def __call__(self, state: dict) -> dict:
        print("Running RAGSemanticChunker...")
        doc: Document = state["document"]
        chunks = self.chunker.split_documents([doc])
        print("finished chunking")

        embeddings = torch.tensor(self.chunker.embeddings.embed_documents(
            [chunk.page_content for chunk in chunks]
        ))
        print(f"Generated {len(chunks)} chunks for document with URL: {doc.metadata.get('url', 'unknown_url')}")

        return {**state, "chunks": chunks, "embeddings": embeddings}
    


class IndexState(TypedDict):
    document: Document # The input document to be indexed
    chunks: List[Document] # The semantic chunks extracted from the document
    embeddings: List[torch.tensor] # The embeddings of the semantic chunks


class IndexingGraph:
    def __init__(
        self
    ):
        self.rag_vectorstore = RAGVectorStore()
        self.embedder = self.rag_vectorstore.embedder
        self.vectorstore = self.rag_vectorstore.vectorstore # Initialize the vectorstore with Weaviate

        self.semantic_chunker = RAGSemanticChunker(
            embedder=self.embedder,
            breakpoint_threshold_type=config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE,
            breakpoint_threshold_amount=config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT
        )

        self.graph = self._build_graph()



    def _store_embeddings(self, state: dict) -> dict:
        self.vectorstore.add_documents(state["chunks"], embeddings=state["embeddings"]) # use precomputed embeddings to bypass the embedder in weaviate
        return state

    def _build_graph(self):
        graph = StateGraph(IndexState)

        graph.add_node("semantic_chunk", self.semantic_chunker)
        graph.add_node("store", RunnableLambda(self._store_embeddings))

        graph.set_entry_point("semantic_chunk")
        graph.add_edge("semantic_chunk", "store")
        graph.set_finish_point("store")

        return graph.compile()

    def run(self, document: Document):
        """Index a single web page Document with 'url' and 'domain' metadata."""
        if "url" not in document.metadata or "domain" not in document.metadata:
            raise ValueError("Document metadata must include 'url' and 'domain'.")

        initial_state = {"document": document}
        return self.graph.invoke(initial_state) # synchronous invocation of the graph for each page

    def delete_by_url(self, url: str, domain: str):
        if self.rag_vectorstore.collections.data.exists(where=Filter.by_property("url").equal(url)):
            print(f"Deleting old document for {url}")
            self.rag_vectorstore.collections.data.delete_many(where=Filter.by_property("url").equal(url))

    def insert_page(self, url: str, domain: str, content: str):
        """Helper method to create a Document and run the indexing graph."""
        doc = Document(page_content=content, metadata={"url": url, "domain": domain})
        self.run(doc)


