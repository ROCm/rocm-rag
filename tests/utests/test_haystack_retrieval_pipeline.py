# test_haystack_retrieval_pipeline.py
import pytest
import os
import sys
from unittest.mock import MagicMock, patch, Mock
from haystack import Document, Pipeline
from typing import List, Dict, Any, Generator

# Add the pipeline wrapper module to path due to hyphen in directory name
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../rocm_rag/retrieval/haystack/ROCm-RAG-Haystack'))
import pipeline_wrapper
from pipeline_wrapper import (
    LoggingWeaviateEmbeddingRetriever,
    PromptBuilderWithRefLinks,
    PipelineWrapper
)

from rocm_rag import config


@pytest.fixture
def mock_document_store():
    """Mock WeaviateDocumentStore"""
    store = MagicMock()
    return store


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        Document(
            content="ROCm is AMD's open-source software platform for GPU computing.",
            meta={"url": "https://rocm.docs.amd.com/en/latest/", "domain": "rocm.docs.amd.com"}
        ),
        Document(
            content="HIP is a C++ runtime API that allows developers to create portable applications.",
            meta={"url": "https://rocm.docs.amd.com/en/latest/reference/hip_runtime_api/", "domain": "rocm.docs.amd.com"}
        ),
        Document(
            content="ROCm supports machine learning frameworks like PyTorch and TensorFlow.",
            meta={"url": "https://rocm.docs.amd.com/en/latest/how-to/deep-learning-rocm/", "domain": "rocm.docs.amd.com"}
        )
    ]


@pytest.fixture
def sample_query_embedding():
    """Sample query embedding for testing"""
    return [0.1] * 1536  # Typical embedding dimension


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is ROCm?"},
        {"role": "assistant", "content": "ROCm is AMD's open-source software platform for GPU computing."},
        {"role": "user", "content": "How does HIP work?"}
    ]


class TestLoggingWeaviateEmbeddingRetriever:
    """Test cases for LoggingWeaviateEmbeddingRetriever"""

    def test_init(self, mock_document_store):
        """Test retriever initialization"""
        retriever = LoggingWeaviateEmbeddingRetriever(document_store=mock_document_store)
        assert retriever._document_store == mock_document_store

    @patch.object(config, 'ROCM_RAG_HAYSTACK_TOP_K_RANKING', 5)
    @patch.object(config, 'ROCM_RAG_HAYSTACK_CERTAINTY_THRESHOLD', 0.8)
    def test_run_with_defaults(self, mock_document_store, sample_documents, sample_query_embedding):
        """Test retriever run method with default parameters"""
        retriever = LoggingWeaviateEmbeddingRetriever(document_store=mock_document_store)
        
        # Mock the parent class run method
        with patch('pipeline_wrapper.WeaviateEmbeddingRetriever.run') as mock_parent_run:
            mock_parent_run.return_value = {"documents": sample_documents}
            
            result = retriever.run(query_embedding=sample_query_embedding)
            
            # Verify parent method was called with correct parameters
            mock_parent_run.assert_called_once_with(
                query_embedding=sample_query_embedding,
                filters=None,
                top_k=5,  # config.ROCM_RAG_HAYSTACK_TOP_K_RANKING
                distance=None,
                certainty=0.8  # config.ROCM_RAG_HAYSTACK_CERTAINTY_THRESHOLD
            )
            
            # Verify result structure
            assert "documents" in result
            assert result["documents"] == sample_documents

    def test_run_with_custom_parameters(self, mock_document_store, sample_documents, sample_query_embedding):
        """Test retriever run method with custom parameters"""
        retriever = LoggingWeaviateEmbeddingRetriever(document_store=mock_document_store)
        
        custom_filters = {"domain": "rocm.docs.amd.com"}
        custom_top_k = 3
        custom_certainty = 0.9
        custom_distance = 0.5
        
        with patch('pipeline_wrapper.WeaviateEmbeddingRetriever.run') as mock_parent_run:
            mock_parent_run.return_value = {"documents": sample_documents[:2]}
            
            result = retriever.run(
                query_embedding=sample_query_embedding,
                filters=custom_filters,
                top_k=custom_top_k,
                distance=custom_distance,
                certainty=custom_certainty
            )
            
            # Verify parent method was called with custom parameters
            mock_parent_run.assert_called_once_with(
                query_embedding=sample_query_embedding,
                filters=custom_filters,
                top_k=custom_top_k,
                distance=custom_distance,
                certainty=custom_certainty
            )
            
            assert len(result["documents"]) == 2


class TestPromptBuilderWithRefLinks:
    """Test cases for PromptBuilderWithRefLinks"""

    def test_run_with_documents(self, sample_documents):
        """Test prompt builder with documents containing URLs"""
        template = "Context: {% for doc in documents %}{{ doc.content }}{% endfor %}\nURLs: {% for url in urls %}{{ url }}{% endfor %}"
        
        prompt_builder = PromptBuilderWithRefLinks(template=template)
        
        # Mock the parent class run method
        with patch('pipeline_wrapper.PromptBuilder.run') as mock_parent_run:
            expected_prompt = "Context: Test content\nURLs: https://example.com"
            mock_parent_run.return_value = {"prompt": expected_prompt}
            
            result = prompt_builder.run(documents=sample_documents)
            
            # Verify URLs were extracted and added to kwargs
            expected_urls = list(set([doc.to_dict()["url"] for doc in sample_documents]))
            
            # Check that parent run was called with URLs added
            call_args = mock_parent_run.call_args
            assert "urls" in call_args[1]
            assert set(call_args[1]["urls"]) == set(expected_urls)
            
            assert result["prompt"] == expected_prompt

    def test_run_without_documents(self):
        """Test prompt builder without documents"""
        template = "No context available"
        prompt_builder = PromptBuilderWithRefLinks(template=template)
        
        with patch('pipeline_wrapper.PromptBuilder.run') as mock_parent_run:
            mock_parent_run.return_value = {"prompt": "No context available"}
            
            result = prompt_builder.run()
            
            # Verify empty URLs list was added
            call_args = mock_parent_run.call_args
            assert call_args[1]["urls"] == []

    def test_run_with_duplicate_urls(self, sample_documents):
        """Test prompt builder handles duplicate URLs correctly"""
        # Create documents with duplicate URLs
        duplicate_docs = [
            Document(content="Content 1", meta={"url": "https://example.com"}),
            Document(content="Content 2", meta={"url": "https://example.com"}),
            Document(content="Content 3", meta={"url": "https://other.com"})
        ]
        
        prompt_builder = PromptBuilderWithRefLinks(template="test template")
        
        with patch('pipeline_wrapper.PromptBuilder.run') as mock_parent_run:
            mock_parent_run.return_value = {"prompt": "test"}
            
            prompt_builder.run(documents=duplicate_docs)
            
            call_args = mock_parent_run.call_args
            urls = call_args[1]["urls"]
            
            # Should have unique URLs only
            assert len(urls) == 2
            assert set(urls) == {"https://example.com", "https://other.com"}


class TestPipelineWrapper:
    """Test cases for PipelineWrapper"""

    @patch('pipeline_wrapper.WeaviateDocumentStore')
    @patch('pipeline_wrapper.OpenAITextEmbedder')
    @patch('pipeline_wrapper.OpenAIGenerator')
    @patch('pipeline_wrapper.Pipeline')
    def test_setup(self, mock_pipeline_class, mock_generator_class, mock_embedder_class, mock_store_class):
        """Test pipeline wrapper setup"""
        # Mock config values
        with patch.multiple(config,
                          ROCM_RAG_WEAVIATE_URL="http://test-weaviate",
                          ROCM_RAG_WEAVIATE_PORT=8080,
                          ROCM_RAG_EMBEDDER_API_BASE_URL="http://test-embedder",
                          ROCM_RAG_EMBEDDER_API_PORT=9090,
                          ROCM_RAG_EMBEDDER_MODEL="test-embedder-model",
                          ROCM_RAG_LLM_API_BASE_URL="http://test-llm",
                          ROCM_RAG_LLM_API_PORT=7070,
                          ROCM_RAG_LLM_MODEL="test-llm-model",
                          ROCM_RAG_WEAVIATE_CLASSNAME="TestClass"):
            
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline
            
            wrapper = PipelineWrapper()
            wrapper.setup()
            
            # Verify WeaviateDocumentStore was created with correct URL
            mock_store_class.assert_called_once()
            store_call_args = mock_store_class.call_args
            assert store_call_args[1]["url"] == "http://test-weaviate:8080"
            
            # Verify OpenAITextEmbedder was created with correct parameters
            mock_embedder_class.assert_called_once_with(
                api_base_url="http://test-embedder:9090/v1",
                model="test-embedder-model"
            )
            
            # Verify OpenAIGenerator was created with correct parameters
            mock_generator_class.assert_called_once_with(
                model="test-llm-model",
                api_base_url="http://test-llm:7070/v1"
            )
            
            # Verify pipeline components were added and connected
            assert mock_pipeline.add_component.call_count == 4
            assert mock_pipeline.connect.call_count == 3
            
            # Verify pipeline was stored
            assert wrapper.pipeline == mock_pipeline

    def test_run_chat_completion_non_streaming(self, sample_messages):
        """Test non-streaming chat completion"""
        wrapper = PipelineWrapper()
        wrapper.pipeline = MagicMock()
        
        # Mock pipeline run result
        mock_result = {
            "llm": {
                "replies": ["This is a test response about ROCm."]
            }
        }
        wrapper.pipeline.run.return_value = mock_result
        
        body = {"stream": False}
        result = wrapper.run_chat_completion("test-model", sample_messages, body)
        
        # Verify pipeline was called with correct arguments
        wrapper.pipeline.run.assert_called_once()
        call_args = wrapper.pipeline.run.call_args[0][0]
        
        assert "text_embedder" in call_args
        assert call_args["text_embedder"]["text"] == "How does HIP work?"  # Last user message
        
        assert "prompt_builder" in call_args
        assert call_args["prompt_builder"]["query"] == sample_messages
        
        assert result == "This is a test response about ROCm."

    def test_run_chat_completion_streaming(self, sample_messages):
        """Test streaming chat completion"""
        wrapper = PipelineWrapper()
        wrapper.pipeline = MagicMock()
        
        with patch('pipeline_wrapper.streaming_generator') as mock_streaming:
            mock_generator = MagicMock()
            mock_streaming.return_value = mock_generator
            
            body = {"stream": True}
            result = wrapper.run_chat_completion("test-model", sample_messages, body)
            
            # Verify streaming_generator was called
            mock_streaming.assert_called_once_with(
                pipeline=wrapper.pipeline,
                pipeline_run_args={
                    "text_embedder": {"text": "How does HIP work?"},
                    "prompt_builder": {"query": sample_messages},
                    "llm": {"generation_kwargs": {}}
                }
            )
            
            assert result == mock_generator

    def test_run_chat_completion_no_replies(self, sample_messages):
        """Test chat completion when no replies are generated"""
        wrapper = PipelineWrapper()
        wrapper.pipeline = MagicMock()
        
        # Mock pipeline run result with no replies
        mock_result = {
            "llm": {
                "replies": []
            }
        }
        wrapper.pipeline.run.return_value = mock_result
        
        body = {"stream": False}
        result = wrapper.run_chat_completion("test-model", sample_messages, body)
        
        assert result == ""

    def test_run_chat_completion_missing_llm_key(self, sample_messages):
        """Test chat completion when LLM key is missing from result"""
        wrapper = PipelineWrapper()
        wrapper.pipeline = MagicMock()
        
        # Mock pipeline run result without llm key
        mock_result = {}
        wrapper.pipeline.run.return_value = mock_result
        
        body = {"stream": False}
        result = wrapper.run_chat_completion("test-model", sample_messages, body)
        
        assert result == ""


class TestIntegration:
    """Integration tests for the complete pipeline"""

    def test_end_to_end_pipeline_flow(self, sample_messages, sample_documents):
        """Test complete end-to-end pipeline flow"""
        wrapper = PipelineWrapper()
        
        # Mock the setup method to avoid actual component initialization
        with patch.object(wrapper, 'setup') as mock_setup:
            mock_pipeline = MagicMock()
            wrapper.pipeline = mock_pipeline
            
            # Mock the pipeline run to simulate actual execution
            mock_pipeline.run.return_value = {
                "llm": {
                    "replies": ["ROCm is AMD's open-source platform for GPU computing. Reference Links:\n- https://rocm.docs.amd.com/en/latest/"]
                }
            }
            
            body = {"stream": False}
            result = wrapper.run_chat_completion("test-model", sample_messages, body)
            
            # Verify the pipeline was executed
            mock_pipeline.run.assert_called_once()
            
            # Verify the response contains expected content
            assert "ROCm" in result
            assert "Reference Links" in result
            assert "https://rocm.docs.amd.com" in result

    def test_prompt_template_rendering(self, sample_documents, sample_messages):
        """Test that the prompt template renders correctly with documents and URLs"""
        prompt_builder = PromptBuilderWithRefLinks(template="test template")
        
        # Test template from pipeline_wrapper.py
        template = """
        Given the following information, answer the question.

        Context: 
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        {% if urls|length != 0 %}
            Reference Links:
            {% for url in urls %}
                - {{ url }}
            {% endfor %}
            Append reference links to end of answer.
        {% endif %}

        History:
        {% for message in query[:-1] %}
            {{ message['role'] }}: {{ message['content'] }}
        {% endfor %}

        Question: {{ query[-1] }}?
        """
        
        with patch('pipeline_wrapper.PromptBuilder.run') as mock_parent_run:
            # Simulate what the parent class would do with the template
            mock_parent_run.return_value = {"prompt": "Rendered prompt with context and URLs"}
            
            result = prompt_builder.run(
                template=template,
                documents=sample_documents,
                query=sample_messages
            )
            
            # Verify URLs were extracted and passed correctly
            call_args = mock_parent_run.call_args
            assert "urls" in call_args[1]
            expected_urls = list(set([doc.to_dict()["url"] for doc in sample_documents]))
            assert set(call_args[1]["urls"]) == set(expected_urls)
            
            # Verify documents were passed through
            assert call_args[1]["documents"] == sample_documents
            assert call_args[1]["query"] == sample_messages
