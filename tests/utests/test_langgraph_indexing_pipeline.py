import pytest
import torch
from unittest.mock import MagicMock, patch, Mock
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.classes.query import Filter
import weaviate
from weaviate.connect import ConnectionParams

from rocm_rag.extraction.langgraph_extraction.indexing_graph import (
    RAGSemanticChunker,
    IndexState,
    IndexingGraph
)
from rocm_rag.utils.langgraph_vectorstore import RAGVectorStore


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        page_content="This is a test document with some content for semantic chunking. "
                    "It has multiple sentences to enable proper chunking behavior. "
                    "The document should be split into meaningful chunks.",
        metadata={"url": "https://example.com/test", "domain": "example.com"}
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Document(
            page_content="This is a test document with some content for semantic chunking.",
            metadata={"url": "https://example.com/test", "domain": "example.com"}
        ),
        Document(
            page_content="It has multiple sentences to enable proper chunking behavior.",
            metadata={"url": "https://example.com/test", "domain": "example.com"}
        ),
        Document(
            page_content="The document should be split into meaningful chunks.",
            metadata={"url": "https://example.com/test", "domain": "example.com"}
        )
    ]


@pytest.fixture
def mock_embedder():
    """Create a mock OpenAI embedder."""
    embedder = MagicMock(spec=OpenAIEmbeddings)
    embedder.embed_documents.return_value = [
        [0.1, 0.2, 0.3, 0.4],  # First chunk embedding
        [0.5, 0.6, 0.7, 0.8],  # Second chunk embedding
        [0.9, 1.0, 1.1, 1.2]   # Third chunk embedding
    ]
    return embedder


@pytest.fixture
def mock_semantic_chunker():
    """Create a mock semantic chunker."""
    chunker = MagicMock(spec=SemanticChunker)
    return chunker


@pytest.fixture
def mock_vectorstore():
    """Create a mock vectorstore."""
    vectorstore = MagicMock()
    vectorstore.add_documents.return_value = None
    return vectorstore


@pytest.fixture
def mock_rag_vectorstore():
    """Create a mock RAGVectorStore."""
    rag_vectorstore = MagicMock()
    rag_vectorstore.embedder = mock_embedder()
    rag_vectorstore.vectorstore = mock_vectorstore()
    rag_vectorstore.collections.data.exists.return_value = True
    rag_vectorstore.collections.data.delete_many.return_value = None
    return rag_vectorstore


class TestRAGSemanticChunker:
    """Test suite for RAGSemanticChunker class."""

    def test_init_default_params(self, mock_embedder):
        """Test RAGSemanticChunker initialization with default parameters."""
        chunker = RAGSemanticChunker(embedder=mock_embedder)
        
        assert chunker.chunker is not None
        assert isinstance(chunker.chunker, SemanticChunker)

    def test_init_custom_params(self, mock_embedder):
        """Test RAGSemanticChunker initialization with custom parameters."""
        chunker = RAGSemanticChunker(
            embedder=mock_embedder,
            buffer_size=2,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=1.5
        )
        
        assert chunker.chunker is not None
        assert isinstance(chunker.chunker, SemanticChunker)

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.SemanticChunker')
    def test_call_method(self, mock_semantic_chunker_class, mock_embedder, sample_document, sample_chunks):
        """Test the __call__ method of RAGSemanticChunker."""
        # Setup mock
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.split_documents.return_value = sample_chunks
        mock_chunker_instance.embeddings = mock_embedder
        mock_semantic_chunker_class.return_value = mock_chunker_instance
        
        # Create chunker and call it
        chunker = RAGSemanticChunker(embedder=mock_embedder)
        state = {"document": sample_document}
        result = chunker(state)
        
        # Verify the call
        mock_chunker_instance.split_documents.assert_called_once_with([sample_document])
        mock_embedder.embed_documents.assert_called_once()
        
        # Verify result structure
        assert "chunks" in result
        assert "embeddings" in result
        assert "document" in result
        assert result["chunks"] == sample_chunks
        assert isinstance(result["embeddings"], torch.Tensor)

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.SemanticChunker')
    def test_call_method_with_no_url_metadata(self, mock_semantic_chunker_class, mock_embedder, sample_chunks):
        """Test __call__ method when document has no URL metadata."""
        # Setup mock
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.split_documents.return_value = sample_chunks
        mock_chunker_instance.embeddings = mock_embedder
        mock_semantic_chunker_class.return_value = mock_chunker_instance
        
        # Create document without URL
        document = Document(
            page_content="Test content",
            metadata={"domain": "example.com"}
        )
        
        chunker = RAGSemanticChunker(embedder=mock_embedder)
        state = {"document": document}
        result = chunker(state)
        
        # Should still work, just print unknown_url
        assert "chunks" in result
        assert "embeddings" in result


class TestIndexState:
    """Test suite for IndexState TypedDict."""

    def test_index_state_structure(self, sample_document, sample_chunks):
        """Test that IndexState can hold the expected data types."""
        embeddings = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        
        # This would be used as type hints, so we just verify the structure
        state_data = {
            "document": sample_document,
            "chunks": sample_chunks,
            "embeddings": embeddings
        }
        
        assert isinstance(state_data["document"], Document)
        assert isinstance(state_data["chunks"], list)
        assert all(isinstance(chunk, Document) for chunk in state_data["chunks"])
        assert isinstance(state_data["embeddings"], torch.Tensor)


class TestIndexingGraph:
    """Test suite for IndexingGraph class."""

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.RAGVectorStore')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.config')
    def test_init(self, mock_config, mock_rag_vectorstore_class):
        """Test IndexingGraph initialization."""
        # Setup mocks
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE = "percentile"
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT = 50.0
        
        mock_rag_vectorstore = MagicMock()
        mock_rag_vectorstore.embedder = MagicMock(spec=OpenAIEmbeddings)
        mock_rag_vectorstore.vectorstore = MagicMock()
        mock_rag_vectorstore_class.return_value = mock_rag_vectorstore
        
        # Create IndexingGraph
        indexing_graph = IndexingGraph()
        
        # Verify initialization
        assert indexing_graph.rag_vectorstore is not None
        assert indexing_graph.embedder is not None
        assert indexing_graph.vectorstore is not None
        assert indexing_graph.semantic_chunker is not None
        assert indexing_graph.graph is not None

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.RAGVectorStore')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.config')
    def test_store_embeddings(self, mock_config, mock_rag_vectorstore_class):
        """Test the _store_embeddings method."""
        # Setup mocks
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE = "percentile"
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT = 50.0
        
        mock_vectorstore = MagicMock()
        mock_rag_vectorstore = MagicMock()
        mock_rag_vectorstore.embedder = MagicMock(spec=OpenAIEmbeddings)
        mock_rag_vectorstore.vectorstore = mock_vectorstore
        mock_rag_vectorstore_class.return_value = mock_rag_vectorstore
        
        indexing_graph = IndexingGraph()
        
        # Test state
        chunks = [Document(page_content="test", metadata={})]
        embeddings = torch.tensor([[0.1, 0.2]])
        state = {"chunks": chunks, "embeddings": embeddings}
        
        # Call method
        result = indexing_graph._store_embeddings(state)
        
        # Verify
        mock_vectorstore.add_documents.assert_called_once_with(chunks, embeddings=embeddings)
        assert result == state

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.RAGVectorStore')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.config')
    def test_build_graph(self, mock_config, mock_rag_vectorstore_class):
        """Test the _build_graph method."""
        # Setup mocks
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE = "percentile"
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT = 50.0
        
        mock_rag_vectorstore = MagicMock()
        mock_rag_vectorstore.embedder = MagicMock(spec=OpenAIEmbeddings)
        mock_rag_vectorstore.vectorstore = MagicMock()
        mock_rag_vectorstore_class.return_value = mock_rag_vectorstore
        
        indexing_graph = IndexingGraph()
        
        # Verify graph is compiled
        assert indexing_graph.graph is not None
        # The graph should be a compiled StateGraph
        assert hasattr(indexing_graph.graph, 'invoke')

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.RAGVectorStore')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.config')
    def test_run_valid_document(self, mock_config, mock_rag_vectorstore_class, sample_document):
        """Test the run method with a valid document."""
        # Setup mocks
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE = "percentile"
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT = 50.0
        
        mock_rag_vectorstore = MagicMock()
        mock_rag_vectorstore.embedder = MagicMock(spec=OpenAIEmbeddings)
        mock_rag_vectorstore.vectorstore = MagicMock()
        mock_rag_vectorstore_class.return_value = mock_rag_vectorstore
        
        indexing_graph = IndexingGraph()
        
        # Mock the graph invoke method
        expected_result = {"document": sample_document, "chunks": [], "embeddings": torch.tensor([])}
        indexing_graph.graph.invoke = MagicMock(return_value=expected_result)
        
        # Call run method
        result = indexing_graph.run(sample_document)
        
        # Verify
        indexing_graph.graph.invoke.assert_called_once_with({"document": sample_document})
        assert result == expected_result

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.RAGVectorStore')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.config')
    def test_run_invalid_document_no_url(self, mock_config, mock_rag_vectorstore_class):
        """Test the run method with document missing URL metadata."""
        # Setup mocks
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE = "percentile"
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT = 50.0
        
        mock_rag_vectorstore = MagicMock()
        mock_rag_vectorstore.embedder = MagicMock(spec=OpenAIEmbeddings)
        mock_rag_vectorstore.vectorstore = MagicMock()
        mock_rag_vectorstore_class.return_value = mock_rag_vectorstore
        
        indexing_graph = IndexingGraph()
        
        # Create document without URL
        invalid_document = Document(
            page_content="Test content",
            metadata={"domain": "example.com"}
        )
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Document metadata must include 'url' and 'domain'"):
            indexing_graph.run(invalid_document)

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.RAGVectorStore')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.config')
    def test_run_invalid_document_no_domain(self, mock_config, mock_rag_vectorstore_class):
        """Test the run method with document missing domain metadata."""
        # Setup mocks
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE = "percentile"
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT = 50.0
        
        mock_rag_vectorstore = MagicMock()
        mock_rag_vectorstore.embedder = MagicMock(spec=OpenAIEmbeddings)
        mock_rag_vectorstore.vectorstore = MagicMock()
        mock_rag_vectorstore_class.return_value = mock_rag_vectorstore
        
        indexing_graph = IndexingGraph()
        
        # Create document without domain
        invalid_document = Document(
            page_content="Test content",
            metadata={"url": "https://example.com/test"}
        )
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Document metadata must include 'url' and 'domain'"):
            indexing_graph.run(invalid_document)

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.Filter')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.RAGVectorStore')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.config')
    def test_delete_by_url_exists(self, mock_config, mock_rag_vectorstore_class, mock_filter):
        """Test delete_by_url when document exists."""
        # Setup mocks
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE = "percentile"
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT = 50.0
        
        mock_collections = MagicMock()
        mock_collections.data.exists.return_value = True
        mock_collections.data.delete_many.return_value = None
        
        mock_rag_vectorstore = MagicMock()
        mock_rag_vectorstore.embedder = MagicMock(spec=OpenAIEmbeddings)
        mock_rag_vectorstore.vectorstore = MagicMock()
        mock_rag_vectorstore.collections = mock_collections
        mock_rag_vectorstore_class.return_value = mock_rag_vectorstore
        
        mock_filter_instance = MagicMock()
        mock_filter.by_property.return_value.equal.return_value = mock_filter_instance
        
        indexing_graph = IndexingGraph()
        
        # Call method
        indexing_graph.delete_by_url("https://example.com/test", "example.com")
        
        # Verify
        mock_collections.data.exists.assert_called_once()
        mock_collections.data.delete_many.assert_called_once()

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.Filter')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.RAGVectorStore')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.config')
    def test_delete_by_url_not_exists(self, mock_config, mock_rag_vectorstore_class, mock_filter):
        """Test delete_by_url when document doesn't exist."""
        # Setup mocks
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE = "percentile"
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT = 50.0
        
        mock_collections = MagicMock()
        mock_collections.data.exists.return_value = False
        
        mock_rag_vectorstore = MagicMock()
        mock_rag_vectorstore.embedder = MagicMock(spec=OpenAIEmbeddings)
        mock_rag_vectorstore.vectorstore = MagicMock()
        mock_rag_vectorstore.collections = mock_collections
        mock_rag_vectorstore_class.return_value = mock_rag_vectorstore
        
        mock_filter_instance = MagicMock()
        mock_filter.by_property.return_value.equal.return_value = mock_filter_instance
        
        indexing_graph = IndexingGraph()
        
        # Call method
        indexing_graph.delete_by_url("https://example.com/test", "example.com")
        
        # Verify exists is called but delete_many is not
        mock_collections.data.exists.assert_called_once()
        mock_collections.data.delete_many.assert_not_called()

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.RAGVectorStore')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.config')
    def test_insert_page(self, mock_config, mock_rag_vectorstore_class):
        """Test the insert_page helper method."""
        # Setup mocks
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE = "percentile"
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT = 50.0
        
        mock_rag_vectorstore = MagicMock()
        mock_rag_vectorstore.embedder = MagicMock(spec=OpenAIEmbeddings)
        mock_rag_vectorstore.vectorstore = MagicMock()
        mock_rag_vectorstore_class.return_value = mock_rag_vectorstore
        
        indexing_graph = IndexingGraph()
        
        # Mock the run method
        indexing_graph.run = MagicMock()
        
        # Call insert_page
        url = "https://example.com/test"
        domain = "example.com"
        content = "Test page content"
        
        indexing_graph.insert_page(url, domain, content)
        
        # Verify that run was called with correct Document
        indexing_graph.run.assert_called_once()
        call_args = indexing_graph.run.call_args[0][0]
        assert isinstance(call_args, Document)
        assert call_args.page_content == content
        assert call_args.metadata["url"] == url
        assert call_args.metadata["domain"] == domain


class TestIntegration:
    """Integration tests for the complete indexing workflow."""

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.RAGVectorStore')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.config')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.SemanticChunker')
    def test_complete_indexing_workflow(self, mock_semantic_chunker_class, mock_config, mock_rag_vectorstore_class, sample_document, sample_chunks):
        """Test the complete indexing workflow from document to storage."""
        # Setup mocks
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE = "percentile"
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT = 50.0
        
        mock_embedder = MagicMock(spec=OpenAIEmbeddings)
        mock_embedder.embed_documents.return_value = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ]
        
        mock_vectorstore = MagicMock()
        mock_rag_vectorstore = MagicMock()
        mock_rag_vectorstore.embedder = mock_embedder
        mock_rag_vectorstore.vectorstore = mock_vectorstore
        mock_rag_vectorstore_class.return_value = mock_rag_vectorstore
        
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.split_documents.return_value = sample_chunks
        mock_chunker_instance.embeddings = mock_embedder
        mock_semantic_chunker_class.return_value = mock_chunker_instance
        
        # Create and run indexing graph
        indexing_graph = IndexingGraph()
        result = indexing_graph.run(sample_document)
        
        # Verify the workflow
        assert "document" in result
        assert "chunks" in result
        assert "embeddings" in result
        
        # Verify vectorstore was called
        mock_vectorstore.add_documents.assert_called_once()
        
        # Verify embeddings were generated
        mock_embedder.embed_documents.assert_called_once()

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.RAGVectorStore')
    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.config')
    def test_error_handling_in_workflow(self, mock_config, mock_rag_vectorstore_class):
        """Test error handling in the indexing workflow."""
        # Setup mocks
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE = "percentile"
        mock_config.ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT = 50.0
        
        mock_rag_vectorstore = MagicMock()
        mock_rag_vectorstore.embedder = MagicMock(spec=OpenAIEmbeddings)
        mock_rag_vectorstore.vectorstore = MagicMock()
        mock_rag_vectorstore_class.return_value = mock_rag_vectorstore
        
        indexing_graph = IndexingGraph()
        
        # Test with invalid document (missing metadata)
        invalid_doc = Document(page_content="Test", metadata={})
        
        with pytest.raises(ValueError):
            indexing_graph.run(invalid_doc)


# Additional test for edge cases and error conditions
class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.SemanticChunker')
    def test_empty_document_chunking(self, mock_semantic_chunker_class, mock_embedder):
        """Test chunking behavior with empty document."""
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.split_documents.return_value = []
        mock_chunker_instance.embeddings = mock_embedder
        mock_semantic_chunker_class.return_value = mock_chunker_instance
        
        chunker = RAGSemanticChunker(embedder=mock_embedder)
        
        empty_doc = Document(page_content="", metadata={"url": "test", "domain": "test"})
        state = {"document": empty_doc}
        
        result = chunker(state)
        
        assert result["chunks"] == []
        assert isinstance(result["embeddings"], torch.Tensor)

    @patch('rocm_rag.extraction.langgraph_extraction.indexing_graph.SemanticChunker')
    def test_large_document_chunking(self, mock_semantic_chunker_class, mock_embedder):
        """Test chunking behavior with very large document."""
        # Create a large document
        large_content = "This is a sentence. " * 1000
        large_doc = Document(
            page_content=large_content,
            metadata={"url": "https://example.com/large", "domain": "example.com"}
        )
        
        # Mock many chunks
        many_chunks = [
            Document(page_content=f"Chunk {i}", metadata=large_doc.metadata)
            for i in range(50)
        ]
        
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.split_documents.return_value = many_chunks
        mock_chunker_instance.embeddings = mock_embedder
        mock_semantic_chunker_class.return_value = mock_chunker_instance
        
        # Mock embeddings for many chunks
        mock_embedder.embed_documents.return_value = [[0.1] * 4 for _ in range(50)]
        
        chunker = RAGSemanticChunker(embedder=mock_embedder)
        state = {"document": large_doc}
        
        result = chunker(state)
        
        assert len(result["chunks"]) == 50
        assert result["embeddings"].shape[0] == 50


class TestRAGVectorStore:
    """Test suite for RAGVectorStore class."""

    @patch('rocm_rag.utils.langgraph_vectorstore.config')
    def test_init_configuration(self, mock_config):
        """Test RAGVectorStore initialization with configuration values."""
        # Setup mock config values
        mock_config.ROCM_RAG_EMBEDDER_MODEL = "text-embedding-3-small"
        mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://localhost"
        mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8080
        mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
        mock_config.ROCM_RAG_WEAVIATE_PORT = 8081
        mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
        mock_config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING = 5
        
        rag_vectorstore = RAGVectorStore()
        
        # Verify configuration is loaded correctly
        assert rag_vectorstore.embedder_model == "text-embedding-3-small"
        assert rag_vectorstore.embedder_api_base_url == "http://localhost:8080/v1"
        assert rag_vectorstore.weaviate_url == "http://localhost:8081"
        assert rag_vectorstore.weaviate_classname == "TestClass"
        assert rag_vectorstore.top_k == 5
        
        # Verify lazy initialization - properties should be None initially
        assert rag_vectorstore._embedder is None
        assert rag_vectorstore._vectorstore is None
        assert rag_vectorstore._collections is None
        assert rag_vectorstore._client is None
        assert rag_vectorstore._retriever is None

    @patch('rocm_rag.utils.langgraph_vectorstore.config')
    @patch('rocm_rag.utils.langgraph_vectorstore.OpenAIEmbeddings')
    def test_embedder_property_lazy_initialization(self, mock_openai_embeddings, mock_config):
        """Test embedder property lazy initialization."""
        # Setup mocks
        mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
        mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://test"
        mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8080
        mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
        mock_config.ROCM_RAG_WEAVIATE_PORT = 8081
        mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
        mock_config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING = 5
        
        mock_embedder_instance = MagicMock(spec=OpenAIEmbeddings)
        mock_openai_embeddings.return_value = mock_embedder_instance
        
        rag_vectorstore = RAGVectorStore()
        
        # First access should initialize the embedder
        embedder = rag_vectorstore.embedder
        
        # Verify initialization
        mock_openai_embeddings.assert_called_once_with(
            model="test-model",
            base_url="http://test:8080/v1",
            tiktoken_enabled=False,
        )
        assert embedder == mock_embedder_instance
        assert rag_vectorstore._embedder == mock_embedder_instance
        
        # Second access should return cached instance
        embedder2 = rag_vectorstore.embedder
        assert embedder2 == mock_embedder_instance
        # Should not call OpenAIEmbeddings again
        mock_openai_embeddings.assert_called_once()

    @patch('rocm_rag.utils.langgraph_vectorstore.config')
    @patch('rocm_rag.utils.langgraph_vectorstore.wait_for_port')
    @patch('rocm_rag.utils.langgraph_vectorstore.ConnectionParams')
    @patch('rocm_rag.utils.langgraph_vectorstore.weaviate')
    def test_client_property_lazy_initialization(self, mock_weaviate, mock_connection_params, 
                                                mock_wait_for_port, mock_config):
        """Test client property lazy initialization with port waiting."""
        # Setup mocks
        mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
        mock_config.ROCM_RAG_WEAVIATE_PORT = 8081
        mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
        mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
        mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://test"
        mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8080
        mock_config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING = 5
        
        mock_connection_params_instance = MagicMock()
        mock_connection_params.from_url.return_value = mock_connection_params_instance
        
        mock_client_instance = MagicMock()
        mock_weaviate.WeaviateClient.return_value = mock_client_instance
        
        rag_vectorstore = RAGVectorStore()
        
        # First access should initialize the client
        client = rag_vectorstore.client
        
        # Verify port waiting was called
        mock_wait_for_port.assert_called_once_with("localhost", 8081)
        
        # Verify connection params creation
        mock_connection_params.from_url.assert_called_once_with(
            url="http://localhost:8081",
            grpc_port=50051,
        )
        
        # Verify client creation and connection
        mock_weaviate.WeaviateClient.assert_called_once_with(
            connection_params=mock_connection_params_instance
        )
        mock_client_instance.connect.assert_called_once()
        
        assert client == mock_client_instance
        assert rag_vectorstore._client == mock_client_instance
        
        # Second access should return cached instance
        client2 = rag_vectorstore.client
        assert client2 == mock_client_instance
        # Should not call initialization again
        mock_wait_for_port.assert_called_once()

    @patch('rocm_rag.utils.langgraph_vectorstore.config')
    @patch('rocm_rag.utils.langgraph_vectorstore.WeaviateVectorStore')
    def test_vectorstore_property_lazy_initialization(self, mock_weaviate_vectorstore, mock_config):
        """Test vectorstore property lazy initialization."""
        # Setup mocks
        mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
        mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
        mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://test"
        mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8080
        mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
        mock_config.ROCM_RAG_WEAVIATE_PORT = 8081
        mock_config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING = 5
        
        mock_vectorstore_instance = MagicMock(spec=WeaviateVectorStore)
        mock_weaviate_vectorstore.return_value = mock_vectorstore_instance
        
        rag_vectorstore = RAGVectorStore()
        
        # Mock the client and embedder properties to avoid their initialization
        mock_client = MagicMock()
        mock_embedder = MagicMock()
        rag_vectorstore._client = mock_client
        rag_vectorstore._embedder = mock_embedder
        
        # First access should initialize the vectorstore
        vectorstore = rag_vectorstore.vectorstore
        
        # Verify initialization
        mock_weaviate_vectorstore.assert_called_once_with(
            client=mock_client,
            index_name="TestClass",
            text_key="content",
            embedding=mock_embedder,
        )
        assert vectorstore == mock_vectorstore_instance
        assert rag_vectorstore._vectorstore == mock_vectorstore_instance
        
        # Second access should return cached instance
        vectorstore2 = rag_vectorstore.vectorstore
        assert vectorstore2 == mock_vectorstore_instance
        # Should not call WeaviateVectorStore again
        mock_weaviate_vectorstore.assert_called_once()

    @patch('rocm_rag.utils.langgraph_vectorstore.config')
    def test_collections_property_lazy_initialization(self, mock_config):
        """Test collections property lazy initialization."""
        # Setup mocks
        mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
        mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
        mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://test"
        mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8080
        mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
        mock_config.ROCM_RAG_WEAVIATE_PORT = 8081
        mock_config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING = 5
        
        rag_vectorstore = RAGVectorStore()
        
        # Mock the client property to avoid its initialization
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_client.collections.get.return_value = mock_collections
        rag_vectorstore._client = mock_client
        
        # First access should initialize the collections
        collections = rag_vectorstore.collections
        
        # Verify initialization
        mock_client.collections.get.assert_called_once_with("TestClass")
        assert collections == mock_collections
        assert rag_vectorstore._collections == mock_collections
        
        # Second access should return cached instance
        collections2 = rag_vectorstore.collections
        assert collections2 == mock_collections
        # Should not call client.collections.get again
        mock_client.collections.get.assert_called_once()

    @patch('rocm_rag.utils.langgraph_vectorstore.config')
    def test_retriever_property_lazy_initialization(self, mock_config):
        """Test retriever property lazy initialization."""
        # Setup mocks
        mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
        mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
        mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://test"
        mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8080
        mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
        mock_config.ROCM_RAG_WEAVIATE_PORT = 8081
        mock_config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING = 3
        
        rag_vectorstore = RAGVectorStore()
        
        # Mock the vectorstore property to avoid its initialization
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        rag_vectorstore._vectorstore = mock_vectorstore
        
        # First access should initialize the retriever
        retriever = rag_vectorstore.retriever
        
        # Verify initialization with correct search_kwargs
        mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        assert retriever == mock_retriever
        assert rag_vectorstore._retriever == mock_retriever
        
        # Second access should return cached instance
        retriever2 = rag_vectorstore.retriever
        assert retriever2 == mock_retriever
        # Should not call as_retriever again
        mock_vectorstore.as_retriever.assert_called_once()

    @patch('rocm_rag.utils.langgraph_vectorstore.config')
    def test_multiple_property_access_integration(self, mock_config):
        """Test that multiple property accesses work together correctly."""
        # Setup mocks
        mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
        mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
        mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://test"
        mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8080
        mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
        mock_config.ROCM_RAG_WEAVIATE_PORT = 8081
        mock_config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING = 5
        
        rag_vectorstore = RAGVectorStore()
        
        # Mock all dependencies to avoid actual initialization
        mock_embedder = MagicMock()
        mock_client = MagicMock()
        mock_vectorstore = MagicMock()
        mock_collections = MagicMock()
        mock_retriever = MagicMock()
        
        rag_vectorstore._embedder = mock_embedder
        rag_vectorstore._client = mock_client
        rag_vectorstore._vectorstore = mock_vectorstore
        rag_vectorstore._collections = mock_collections
        rag_vectorstore._retriever = mock_retriever
        
        # Access all properties multiple times
        assert rag_vectorstore.embedder == mock_embedder
        assert rag_vectorstore.client == mock_client
        assert rag_vectorstore.vectorstore == mock_vectorstore
        assert rag_vectorstore.collections == mock_collections
        assert rag_vectorstore.retriever == mock_retriever
        
        # Verify caching - second access should return same instances
        assert rag_vectorstore.embedder == mock_embedder
        assert rag_vectorstore.client == mock_client
        assert rag_vectorstore.vectorstore == mock_vectorstore
        assert rag_vectorstore.collections == mock_collections
        assert rag_vectorstore.retriever == mock_retriever


class TestRAGVectorStoreEdgeCases:
    """Test edge cases and error conditions for RAGVectorStore."""

    @patch('rocm_rag.utils.langgraph_vectorstore.config')
    def test_config_with_different_port_types(self, mock_config):
        """Test configuration with different port value types."""
        # Test with string ports (should still work)
        mock_config.ROCM_RAG_EMBEDDER_API_PORT = "8080"
        mock_config.ROCM_RAG_WEAVIATE_PORT = "8081"
        mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
        mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://test"
        mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
        mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
        mock_config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING = 5
        
        rag_vectorstore = RAGVectorStore()
        
        assert rag_vectorstore.embedder_api_base_url == "http://test:8080/v1"
        assert rag_vectorstore.weaviate_url == "http://localhost:8081"

    @patch('rocm_rag.utils.langgraph_vectorstore.config')
    def test_config_with_zero_top_k(self, mock_config):
        """Test configuration with zero top_k value."""
        mock_config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING = 0
        mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
        mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://test"
        mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8080
        mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
        mock_config.ROCM_RAG_WEAVIATE_PORT = 8081
        mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
        
        rag_vectorstore = RAGVectorStore()
        assert rag_vectorstore.top_k == 0

    @patch('rocm_rag.utils.langgraph_vectorstore.config')
    def test_config_with_special_characters_in_classname(self, mock_config):
        """Test configuration with special characters in class name."""
        mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "Test_Class-2024"
        mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
        mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://test"
        mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8080
        mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
        mock_config.ROCM_RAG_WEAVIATE_PORT = 8081
        mock_config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING = 5
        
        rag_vectorstore = RAGVectorStore()
        assert rag_vectorstore.weaviate_classname == "Test_Class-2024"

    @patch('rocm_rag.utils.langgraph_vectorstore.config')
    @patch('rocm_rag.utils.langgraph_vectorstore.wait_for_port')
    def test_client_property_with_wait_for_port_called(self, mock_wait_for_port, mock_config):
        """Test that client property correctly calls wait_for_port."""
        # Setup mocks
        mock_config.ROCM_RAG_WEAVIATE_URL = "http://testhost"
        mock_config.ROCM_RAG_WEAVIATE_PORT = 9999
        mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
        mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
        mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://test"
        mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8080
        mock_config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING = 5
        
        rag_vectorstore = RAGVectorStore()
        
        # Mock the actual client creation to avoid connection attempts
        with patch('rocm_rag.utils.langgraph_vectorstore.ConnectionParams'), \
             patch('rocm_rag.utils.langgraph_vectorstore.weaviate'):
            
            # Access client property
            _ = rag_vectorstore.client
            
            # Verify wait_for_port was called with correct parameters
            mock_wait_for_port.assert_called_once_with("localhost", 9999)

    @patch('rocm_rag.utils.langgraph_vectorstore.config')
    def test_embedder_api_base_url_construction(self, mock_config):
        """Test correct construction of embedder API base URL."""
        mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "https://api.example.com"
        mock_config.ROCM_RAG_EMBEDDER_API_PORT = 443
        mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
        mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
        mock_config.ROCM_RAG_WEAVIATE_PORT = 8081
        mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
        mock_config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING = 5
        
        rag_vectorstore = RAGVectorStore()
        
        assert rag_vectorstore.embedder_api_base_url == "https://api.example.com:443/v1"

    @patch('rocm_rag.utils.langgraph_vectorstore.config')
    def test_weaviate_url_construction(self, mock_config):
        """Test correct construction of Weaviate URL."""
        mock_config.ROCM_RAG_WEAVIATE_URL = "https://weaviate.example.com"
        mock_config.ROCM_RAG_WEAVIATE_PORT = 443
        mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
        mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://test"
        mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8080
        mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
        mock_config.ROCM_RAG_LANGGRAPH_TOP_K_RANKING = 5
        
        rag_vectorstore = RAGVectorStore()
        
        assert rag_vectorstore.weaviate_url == "https://weaviate.example.com:443"
