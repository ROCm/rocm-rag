import weaviate
from weaviate.connect import ConnectionParams
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
from rocm_rag import config
from rocm_rag.utils.wait_for_service import wait_for_port


class RAGVectorStore:
    """
    Encapsulates Weaviate vectorstore and embedding initialization.
    Lazily connects to Weaviate and waits for port when needed.
    """

    def __init__(self):
        self.embedder_model = config.ROCM_RAG_EMBEDDER_MODEL
        self.embedder_api_base_url = f"{config.ROCM_RAG_EMBEDDER_API_BASE_URL}:{config.ROCM_RAG_EMBEDDER_API_PORT}/v1"
        self.weaviate_url = f"{config.ROCM_RAG_WEAVIATE_URL}:{config.ROCM_RAG_WEAVIATE_PORT}"
        self.weaviate_classname = config.ROCM_RAG_WEAVIATE_CLASSNAME

        self._embedder = None
        self._vectorstore = None
        self._collections = None
        self._client = None
        self._retriever = None
        self.top_k = config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING

    @property
    def embedder(self) -> OpenAIEmbeddings:
        """Initialize embedder once."""
        if self._embedder is None:
            self._embedder = OpenAIEmbeddings(
                model=self.embedder_model,
                base_url=self.embedder_api_base_url,
                tiktoken_enabled=False,
            )
        return self._embedder

    @property
    def client(self) -> weaviate.WeaviateClient:
        """Connect to Weaviate once, waiting for port if needed."""
        if self._client is None:
            wait_for_port("localhost", config.ROCM_RAG_WEAVIATE_PORT)

            connection_params = ConnectionParams.from_url(
                url=self.weaviate_url,
                grpc_port=50051,
            )
            self._client = weaviate.WeaviateClient(connection_params=connection_params)
            self._client.connect()
        return self._client

    @property
    def vectorstore(self) -> WeaviateVectorStore:
        """Initialize vectorstore once."""
        if self._vectorstore is None:
            self._vectorstore = WeaviateVectorStore(
                client=self.client,
                index_name=self.weaviate_classname,
                text_key="content",
                embedding=self.embedder,
            )
        return self._vectorstore

    @property
    def collections(self):
        """Retrieve Weaviate collection object once."""
        if self._collections is None:
            self._collections = self.client.collections.get(self.weaviate_classname)
        return self._collections

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        return self._retriever

    
