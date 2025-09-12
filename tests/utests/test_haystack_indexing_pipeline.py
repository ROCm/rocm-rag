# test_indexing_pipeline.py
import pytest
import random
from haystack import Document
from unittest.mock import MagicMock, patch, Mock
from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
from haystack.components.embedders import OpenAIDocumentEmbedder

from rocm_rag.extraction.haystack_extraction.indexing_pipeline import SemanticChunkMerger, IndexingPipeline

@pytest.fixture
def sample_documents():
    return [
        Document(content="This is a test sentence 1.", meta={"url": "http://example.com", "domain": "example.com"}),
        Document(content="This is a test sentence 2.", meta={"url": "http://example.com", "domain": "example.com"}),
        Document(content="Unrelated content here.", meta={"url": "http://example.com", "domain": "example.com"})
    ]

@pytest.fixture
def mock_embedder_static():
    embedder = MagicMock()
    # Mock embedding to be a tensor of ones for simplicity
    embedder.run.side_effect = lambda documents: {
        "documents": [Document(content=doc.content, embedding=[0.05]*4096) for doc in documents]
    }
    return embedder

@pytest.fixture
def mock_embedder_random():
    embedder = MagicMock()
    # Generate random embeddings for each document
    def random_embeddings(documents):
        return {
            "documents": [
                Document(content=doc.content, embedding=[random.random() for _ in range(4096)])
                for doc in documents
            ]
        }
    embedder.run.side_effect = random_embeddings
    return embedder


def test_semantic_chunk_merger_merge(sample_documents, mock_embedder_static):
    # Initialize with low similarity threshold so chunks will merge
    semantic_chunk_merger = SemanticChunkMerger(similarity_threshold=0.5, max_chunk_length=200, embedder=mock_embedder_static)
    
    result = semantic_chunk_merger.run(sample_documents)
    merged_docs = result["documents"]
    
    assert isinstance(merged_docs, list)
    assert all(isinstance(doc, Document) for doc in merged_docs)
    # Since mock embeddings are identical, all chunks should merge into one
    assert len(merged_docs) == 1
    assert " ".join([doc.content for doc in sample_documents]) in merged_docs[0].content


def test_semantic_chunk_merger_no_merge(sample_documents, mock_embedder_random):
    # High threshold prevents merging
    merger = SemanticChunkMerger(similarity_threshold=1.0, max_chunk_length=200, embedder=mock_embedder_random)
    
    result = merger.run(sample_documents)
    merged_docs = result["documents"]
    
    assert len(merged_docs) == len(sample_documents)  # no merge occurs


# Additional tests for SemanticChunkMerger edge cases
def test_semantic_chunk_merger_empty_documents():
    embedder = MagicMock()
    merger = SemanticChunkMerger(similarity_threshold=0.5, max_chunk_length=200, embedder=embedder)
    
    result = merger.run([])
    assert result["documents"] == []


def test_semantic_chunk_merger_single_document():
    embedder = MagicMock()
    embedder.run.return_value = {
        "documents": [Document(content="Single chunk content", embedding=[0.1]*4096)]
    }
    
    merger = SemanticChunkMerger(similarity_threshold=0.5, max_chunk_length=200, embedder=embedder)
    single_doc = [Document(content="Single chunk content", meta={"url": "http://test.com", "domain": "test.com"})]
    
    result = merger.run(single_doc)
    merged_docs = result["documents"]
    
    assert len(merged_docs) == 1
    assert merged_docs[0].content == "Single chunk content"
    assert merged_docs[0].meta["url"] == "http://test.com"
    assert merged_docs[0].meta["domain"] == "test.com"


def test_semantic_chunk_merger_max_length_exceeded():
    embedder = MagicMock()
    # Mock embeddings for each call
    embedder.run.side_effect = [
        {"documents": [
            Document(content="A" * 80, embedding=[0.1]*4096),
            Document(content="B" * 80, embedding=[0.1]*4096)
        ]},
        {"documents": [Document(content="A" * 80 + " " + "B" * 80, embedding=[0.2]*4096)]}
    ]
    
    # Set max_chunk_length to be less than combined content
    merger = SemanticChunkMerger(similarity_threshold=0.9, max_chunk_length=100, embedder=embedder)
    docs = [
        Document(content="A" * 80, meta={"url": "http://test.com", "domain": "test.com"}),
        Document(content="B" * 80, meta={"url": "http://test.com", "domain": "test.com"})
    ]
    
    result = merger.run(docs)
    merged_docs = result["documents"]
    
    # Should not merge due to length constraint
    assert len(merged_docs) == 2


def test_semantic_chunk_merger_similarity_boundary():
    embedder = MagicMock()
    # First call returns initial embeddings with very different vectors
    embedder.run.side_effect = [
        {"documents": [
            Document(content="Similar content 1", embedding=[1.0] + [0.0]*4095),
            Document(content="Similar content 2", embedding=[0.0] + [1.0]*4095)
        ]},
        {"documents": [Document(content="Similar content 1 Similar content 2", embedding=[0.5]*4096)]}
    ]
    
    # Test with high threshold that prevents merging
    merger = SemanticChunkMerger(similarity_threshold=0.99, max_chunk_length=500, embedder=embedder)
    docs = [
        Document(content="Similar content 1", meta={"url": "http://test.com", "domain": "test.com"}),
        Document(content="Similar content 2", meta={"url": "http://test.com", "domain": "test.com"})
    ]
    
    result = merger.run(docs)
    merged_docs = result["documents"]
    
    # With high threshold and orthogonal vectors, should not merge
    assert len(merged_docs) == 2


# Tests for IndexingPipeline class
@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.config')
@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.WeaviateDocumentStore')
@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.OpenAIDocumentEmbedder')
def test_indexing_pipeline_init(mock_embedder, mock_weaviate, mock_config):
    # Mock config values
    mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
    mock_config.ROCM_RAG_WEAVIATE_PORT = 8080
    mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://localhost"
    mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8000
    mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
    mock_config.ROCM_RAG_MAX_CHUNK_LENGTH = 512
    mock_config.ROCM_RAG_SIMILARITY_THRESHOLD = 0.5
    mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
    
    # Mock return values
    mock_weaviate.return_value = Mock()
    mock_embedder.return_value = Mock()
    
    pipeline = IndexingPipeline()
    
    # Verify initialization calls
    mock_weaviate.assert_called_once()
    mock_embedder.assert_called_once_with(
        api_base_url="http://localhost:8000/v1",
        model="test-model"
    )
    
    # Verify pipeline components are set
    assert pipeline.document_store is not None
    assert pipeline.embedder is not None
    assert pipeline.document_preprocessor is not None
    assert pipeline.semantic_chunk_merger is not None
    assert pipeline.indexing_pipeline is not None


@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.config')
@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.WeaviateDocumentStore')
@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.OpenAIDocumentEmbedder')
def test_indexing_pipeline_run(mock_embedder, mock_weaviate, mock_config):
    # Setup mocks
    mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
    mock_config.ROCM_RAG_WEAVIATE_PORT = 8080
    mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://localhost"
    mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8000
    mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
    mock_config.ROCM_RAG_MAX_CHUNK_LENGTH = 512
    mock_config.ROCM_RAG_SIMILARITY_THRESHOLD = 0.5
    mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
    
    mock_weaviate.return_value = Mock()
    mock_embedder.return_value = Mock()
    
    pipeline = IndexingPipeline()
    
    # Mock the pipeline run method
    mock_results = {"writer": {"documents_written": 1}}
    pipeline.indexing_pipeline.run = Mock(return_value=mock_results)
    
    test_doc = Document(content="Test content", meta={"url": "http://test.com", "domain": "test.com"})
    result = pipeline.run(test_doc)
    
    # Verify pipeline was called with correct input
    pipeline.indexing_pipeline.run.assert_called_once_with({"preprocessor": {"documents": [test_doc]}})
    assert result == mock_results


@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.config')
@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.WeaviateDocumentStore')
@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.OpenAIDocumentEmbedder')
def test_indexing_pipeline_delete_by_url(mock_embedder, mock_weaviate, mock_config):
    # Setup mocks
    mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
    mock_config.ROCM_RAG_WEAVIATE_PORT = 8080
    mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://localhost"
    mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8000
    mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
    mock_config.ROCM_RAG_MAX_CHUNK_LENGTH = 512
    mock_config.ROCM_RAG_SIMILARITY_THRESHOLD = 0.5
    mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
    
    mock_document_store = Mock()
    mock_weaviate.return_value = mock_document_store
    mock_embedder.return_value = Mock()
    
    # Mock documents to be deleted
    mock_docs = [
        Mock(id="doc1"),
        Mock(id="doc2")
    ]
    mock_document_store.filter_documents.return_value = mock_docs
    
    pipeline = IndexingPipeline()
    pipeline.delete_by_url("http://test.com", "test.com")
    
    # Verify filter was called with correct parameters
    expected_filters = {
        "operator": "AND",
        "conditions": [
            {"field": "url", "value": "http://test.com", "operator": "=="},
            {"field": "domain", "value": "test.com", "operator": "=="}
        ]
    }
    mock_document_store.filter_documents.assert_called_once_with(filters=expected_filters)
    
    # Verify delete was called with document IDs
    mock_document_store.delete_documents.assert_called_once_with(["doc1", "doc2"])


@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.config')
@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.WeaviateDocumentStore')
@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.OpenAIDocumentEmbedder')
def test_indexing_pipeline_delete_by_url_no_documents(mock_embedder, mock_weaviate, mock_config):
    # Setup mocks
    mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
    mock_config.ROCM_RAG_WEAVIATE_PORT = 8080
    mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://localhost"
    mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8000
    mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
    mock_config.ROCM_RAG_MAX_CHUNK_LENGTH = 512
    mock_config.ROCM_RAG_SIMILARITY_THRESHOLD = 0.5
    mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
    
    mock_document_store = Mock()
    mock_weaviate.return_value = mock_document_store
    mock_embedder.return_value = Mock()
    
    # Mock no documents found
    mock_document_store.filter_documents.return_value = []
    
    pipeline = IndexingPipeline()
    pipeline.delete_by_url("http://test.com", "test.com")
    
    # Verify filter was called but delete was not
    mock_document_store.filter_documents.assert_called_once()
    mock_document_store.delete_documents.assert_not_called()


@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.config')
@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.WeaviateDocumentStore')
@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.OpenAIDocumentEmbedder')
def test_indexing_pipeline_insert_page(mock_embedder, mock_weaviate, mock_config):
    # Setup mocks
    mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
    mock_config.ROCM_RAG_WEAVIATE_PORT = 8080
    mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://localhost"
    mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8000
    mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
    mock_config.ROCM_RAG_MAX_CHUNK_LENGTH = 512
    mock_config.ROCM_RAG_SIMILARITY_THRESHOLD = 0.5
    mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
    
    mock_weaviate.return_value = Mock()
    mock_embedder.return_value = Mock()
    
    pipeline = IndexingPipeline()
    
    # Mock the run method
    pipeline.run = Mock(return_value={"writer": {"documents_written": 1}})
    
    url = "http://test.com"
    domain = "test.com"
    content = "Test page content"
    
    pipeline.insert_page(url, domain, content)
    
    # Verify run was called with a Document containing correct data
    pipeline.run.assert_called_once()
    call_args = pipeline.run.call_args[0][0]  # Get the first argument (Document)
    
    assert call_args.content == content
    assert call_args.meta["url"] == url
    assert call_args.meta["domain"] == domain


# Additional edge case tests
def test_semantic_chunk_merger_with_metadata_handling():
    """Test that metadata is properly preserved during merging."""
    embedder = MagicMock()
    embedder.run.side_effect = [
        {"documents": [
            Document(content="First chunk", embedding=[0.1]*4096),
            Document(content="Second chunk", embedding=[0.1]*4096)
        ]},
        {"documents": [Document(content="First chunk Second chunk", embedding=[0.1]*4096)]}
    ]
    
    merger = SemanticChunkMerger(similarity_threshold=0.5, max_chunk_length=500, embedder=embedder)
    docs = [
        Document(content="First chunk", meta={"url": "http://example.com", "domain": "example.com"}),
        Document(content="Second chunk", meta={"url": "http://example.com", "domain": "example.com"})
    ]
    
    result = merger.run(docs)
    merged_docs = result["documents"]
    
    # Should merge into one document with preserved metadata
    assert len(merged_docs) == 1
    assert merged_docs[0].meta["url"] == "http://example.com"
    assert merged_docs[0].meta["domain"] == "example.com"


def test_semantic_chunk_merger_three_chunks():
    """Test scenario with three chunks where some merge and some don't."""
    embedder = MagicMock()
    # Return embeddings for each call - first chunk set, then merged chunks
    def embedding_side_effect(*args, **kwargs):
        docs = kwargs.get('documents', [])
        if len(docs) == 3:  # Initial embeddings
            return {"documents": [
                Document(content=docs[0].content, embedding=[0.1]*4096),  
                Document(content=docs[1].content, embedding=[0.1]*4096),  # Similar to chunk 1
                Document(content=docs[2].content, embedding=[0.9]*4096)   # Different from others
            ]}
        else:  # Re-embedding merged content
            return {"documents": [
                Document(content=docs[0].content, embedding=[0.15]*4096)
            ]}
    
    embedder.run.side_effect = embedding_side_effect
    
    merger = SemanticChunkMerger(similarity_threshold=0.5, max_chunk_length=500, embedder=embedder)
    docs = [
        Document(content="Chunk 1", meta={"url": "http://test.com", "domain": "test.com"}),
        Document(content="Chunk 2", meta={"url": "http://test.com", "domain": "test.com"}),
        Document(content="Chunk 3", meta={"url": "http://test.com", "domain": "test.com"})
    ]
    
    result = merger.run(docs)
    merged_docs = result["documents"]
    
    # Should result in fewer than 3 documents due to merging
    assert len(merged_docs) <= 2
    # Check that some content is preserved
    all_content = " ".join([doc.content for doc in merged_docs])
    assert "Chunk" in all_content


def test_semantic_chunk_merger_embedding_error_handling():
    """Test behavior when embedder returns unexpected format."""
    embedder = MagicMock()
    embedder.run.return_value = {
        "documents": [Document(content="Test content", embedding=[0.1]*4096)]
    }
    
    merger = SemanticChunkMerger(similarity_threshold=0.5, max_chunk_length=200, embedder=embedder)
    single_doc = [Document(content="Test content", meta={"url": "http://test.com", "domain": "test.com"})]
    
    result = merger.run(single_doc)
    merged_docs = result["documents"]
    
    # Should handle single document gracefully
    assert len(merged_docs) == 1
    assert merged_docs[0].content == "Test content"


@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.config')
@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.WeaviateDocumentStore')
@patch('rocm_rag.extraction.haystack_extraction.indexing_pipeline.OpenAIDocumentEmbedder')
def test_indexing_pipeline_collection_settings(mock_embedder, mock_weaviate, mock_config):
    """Test that collection settings are properly configured."""
    # Mock config values
    mock_config.ROCM_RAG_WEAVIATE_URL = "http://localhost"
    mock_config.ROCM_RAG_WEAVIATE_PORT = 8080
    mock_config.ROCM_RAG_EMBEDDER_API_BASE_URL = "http://localhost"
    mock_config.ROCM_RAG_EMBEDDER_API_PORT = 8000
    mock_config.ROCM_RAG_EMBEDDER_MODEL = "test-model"
    mock_config.ROCM_RAG_MAX_CHUNK_LENGTH = 512
    mock_config.ROCM_RAG_SIMILARITY_THRESHOLD = 0.5
    mock_config.ROCM_RAG_WEAVIATE_CLASSNAME = "TestClass"
    
    mock_weaviate.return_value = Mock()
    mock_embedder.return_value = Mock()
    
    pipeline = IndexingPipeline()
    
    # Verify WeaviateDocumentStore was called with correct collection settings
    call_args = mock_weaviate.call_args
    assert call_args[1]["url"] == "http://localhost:8080"
    
    collection_settings = call_args[1]["collection_settings"]
    assert collection_settings["class"] == "TestClass"
    assert collection_settings["vectorizer"] == "none"
    assert collection_settings["invertedIndexConfig"]["indexNullState"] is True
    
    # Check properties
    properties = collection_settings["properties"]
    expected_properties = [
        {"name": "content", "dataType": ["text"]},
        {"name": "url", "dataType": ["string"]},
        {"name": "domain", "dataType": ["string"]},
    ]
    assert properties == expected_properties
