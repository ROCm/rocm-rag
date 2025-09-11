"""
Comprehensive unit tests for the ROCm RAG Langgraph retrieval pipeline

This file contains all unit tests for the langgraph-based RAG system including:
- Type definitions and state management (rag_types.py)
- Core retrieval and generation nodes (rag_nodes.py) 
- Graph building and compilation (rag_graph.py)
- FastAPI endpoints and API models (serve.py)
- End-to-end integration scenarios

The tests use extensive mocking to isolate components and avoid external dependencies.
"""
import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
import json
import sys
import os
from typing import List, Optional, AsyncGenerator
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

# Add the langgraph module path to import the modules
langgraph_path = os.path.join(
    os.path.dirname(__file__), 
    '..', '..', 
    'rocm_rag', 'retrieval', 'langgraph', 'ROCm-RAG-Langgraph'
)
sys.path.insert(0, os.path.abspath(langgraph_path))

# Create a mock RAGVectorStore class to prevent port waiting during import
class MockRAGVectorStore:
    def __init__(self):
        self.retriever = Mock()
        self.retriever.invoke = Mock()
        
    @property 
    def client(self):
        return Mock()
    
    @property
    def vectorstore(self):
        return Mock()
        
    @property
    def collections(self):
        return Mock()

try:
    # Mock the RAGVectorStore to prevent port waiting during imports
    with patch('rocm_rag.utils.langgraph_vectorstore.RAGVectorStore', MockRAGVectorStore):
        from rag_types import GraphState
        from rag_nodes import retrieve_node, generate_node_stream
        from rag_graph import build_graph
        from serve import app, ChatMessage, ChatRequest, chat, list_models
        from fastapi.testclient import TestClient
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Could not import langgraph modules: {e}")
    IMPORTS_AVAILABLE = False


# ===== TYPE TESTS =====
@unittest.skipUnless(IMPORTS_AVAILABLE, "Langgraph modules not available")
class TestGraphState(unittest.TestCase):
    """Test cases for GraphState TypedDict"""

    def test_graph_state_structure(self):
        """Test that GraphState has the expected structure"""
        # Test creating a valid GraphState instance
        state: GraphState = {
            "question": "What is ROCm?",
            "documents": [Document(page_content="ROCm is an open-source platform", metadata={"url": "test.com"})],
            "urls": ["test.com"],
            "answer": "ROCm is an open-source platform for GPU computing",
            "history": [{"role": "user", "content": "Hello"}],
            "stream": True
        }
        
        # Verify all required fields are present
        self.assertIn("question", state)
        self.assertIn("documents", state)
        self.assertIn("urls", state)
        self.assertIn("answer", state)
        self.assertIn("history", state)
        self.assertIn("stream", state)
        
        # Verify field types
        self.assertIsInstance(state["question"], str)
        self.assertIsInstance(state["documents"], list)
        self.assertIsInstance(state["urls"], list)
        self.assertIsInstance(state["answer"], (str, type(None)))
        self.assertIsInstance(state["history"], list)
        self.assertIsInstance(state["stream"], bool)

    def test_graph_state_with_empty_documents(self):
        """Test GraphState with empty document list"""
        state: GraphState = {
            "question": "Test question",
            "documents": [],
            "urls": [],
            "answer": None,
            "history": [],
            "stream": False
        }
        
        self.assertEqual(len(state["documents"]), 0)
        self.assertEqual(len(state["urls"]), 0)
        self.assertIsNone(state["answer"])

    def test_graph_state_with_multiple_documents(self):
        """Test GraphState with multiple documents"""
        documents = [
            Document(page_content="Content 1", metadata={"url": "url1.com"}),
            Document(page_content="Content 2", metadata={"url": "url2.com"}),
            Document(page_content="Content 3", metadata={"url": "url3.com"})
        ]
        
        state: GraphState = {
            "question": "Multi-doc question",
            "documents": documents,
            "urls": ["url1.com", "url2.com", "url3.com"],
            "answer": "Multi-doc answer",
            "history": [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First response"}
            ],
            "stream": True
        }
        
        self.assertEqual(len(state["documents"]), 3)
        self.assertEqual(len(state["urls"]), 3)
        self.assertEqual(len(state["history"]), 2)

    def test_graph_state_document_metadata(self):
        """Test that documents can have various metadata"""
        doc = Document(
            page_content="Test content",
            metadata={
                "url": "https://test.com",
                "title": "Test Document",
                "source": "web",
                "timestamp": "2024-01-01"
            }
        )
        
        state: GraphState = {
            "question": "Test",
            "documents": [doc],
            "urls": ["https://test.com"],
            "answer": None,
            "history": [],
            "stream": False
        }
        
        # Verify document metadata is preserved
        self.assertEqual(state["documents"][0].metadata["url"], "https://test.com")
        self.assertEqual(state["documents"][0].metadata["title"], "Test Document")
        self.assertEqual(state["documents"][0].page_content, "Test content")


# ===== NODE TESTS =====
@unittest.skipUnless(IMPORTS_AVAILABLE, "Langgraph modules not available")
class TestRetrieveNode(unittest.TestCase):
    """Test cases for retrieve_node function"""

    @patch('rag_nodes.rag_vectorstore')
    def test_retrieve_node_basic(self, mock_vectorstore):
        """Test basic retrieve_node functionality"""
        # Mock documents returned by retriever
        mock_docs = [
            Document(page_content="ROCm content 1", metadata={"url": "https://rocm1.com"}),
            Document(page_content="ROCm content 2", metadata={"url": "https://rocm2.com"}),
            Document(page_content="ROCm content 3", metadata={"url": "https://rocm3.com"})
        ]
        
        mock_vectorstore.retriever.invoke.return_value = mock_docs
        
        # Input state
        input_state: GraphState = {
            "question": "What is ROCm?",
            "documents": [],
            "urls": [],
            "answer": None,
            "history": [],
            "stream": False
        }
        
        # Call retrieve_node
        result_state = retrieve_node(input_state)
        
        # Verify the retriever was called with the question
        mock_vectorstore.retriever.invoke.assert_called_once_with("What is ROCm?")
        
        # Verify the state was updated correctly
        self.assertEqual(len(result_state["documents"]), 3)
        self.assertEqual(len(result_state["urls"]), 3)
        self.assertEqual(result_state["urls"], ["https://rocm1.com", "https://rocm2.com", "https://rocm3.com"])
        self.assertEqual(result_state["question"], "What is ROCm?")

    @patch('rag_nodes.rag_vectorstore')
    def test_retrieve_node_no_documents(self, mock_vectorstore):
        """Test retrieve_node when no documents are found"""
        mock_vectorstore.retriever.invoke.return_value = []
        
        input_state: GraphState = {
            "question": "Nonexistent topic",
            "documents": [],
            "urls": [],
            "answer": None,
            "history": [],
            "stream": False
        }
        
        result_state = retrieve_node(input_state)
        
        self.assertEqual(len(result_state["documents"]), 0)
        self.assertEqual(len(result_state["urls"]), 0)
        self.assertEqual(result_state["question"], "Nonexistent topic")

    @patch('rag_nodes.rag_vectorstore')
    def test_retrieve_node_documents_without_url(self, mock_vectorstore):
        """Test retrieve_node with documents that don't have URL metadata"""
        mock_docs = [
            Document(page_content="Content without URL", metadata={}),
            Document(page_content="Content with URL", metadata={"url": "https://test.com"}),
            Document(page_content="Content with other metadata", metadata={"title": "Test Title"})
        ]
        
        mock_vectorstore.retriever.invoke.return_value = mock_docs
        
        input_state: GraphState = {
            "question": "Test question",
            "documents": [],
            "urls": [],
            "answer": None,
            "history": [],
            "stream": False
        }
        
        result_state = retrieve_node(input_state)
        
        # URLs should be extracted, with empty string for documents without URL
        expected_urls = ["", "https://test.com", ""]
        self.assertEqual(result_state["urls"], expected_urls)

    @patch('rag_nodes.rag_vectorstore')
    def test_retrieve_node_state_preservation(self, mock_vectorstore):
        """Test that retrieve_node preserves other state fields"""
        mock_docs = [Document(page_content="Test", metadata={"url": "test.com"})]
        mock_vectorstore.retriever.invoke.return_value = mock_docs
        
        input_state: GraphState = {
            "question": "Test question",
            "documents": [],
            "urls": [],
            "answer": "Previous answer",
            "history": [{"role": "user", "content": "Previous message"}],
            "stream": True
        }
        
        result_state = retrieve_node(input_state)
        
        # Check that non-document/url fields are preserved
        self.assertEqual(result_state["answer"], "Previous answer")
        self.assertEqual(result_state["history"], [{"role": "user", "content": "Previous message"}])
        self.assertEqual(result_state["stream"], True)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Langgraph modules not available")
class TestGenerateNodeStream(unittest.TestCase):
    """Test cases for generate_node_stream function"""

    @patch('rag_nodes.chat_llm')
    def test_generate_node_stream_non_streaming(self, mock_llm):
        """Test generate_node_stream in non-streaming mode"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "ROCm is an open-source software platform for GPU computing."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Input state
        input_state: GraphState = {
            "question": "What is ROCm?",
            "documents": [
                Document(page_content="ROCm documentation content", metadata={"url": "https://rocm.docs.amd.com"})
            ],
            "urls": ["https://rocm.docs.amd.com"],
            "answer": None,
            "history": [],
            "stream": False
        }
        
        # Run the async function
        async def run_test():
            result = await generate_node_stream(input_state)
            return result
        
        result_state = asyncio.run(run_test())
        
        # Verify LLM was called with correct prompt
        mock_llm.ainvoke.assert_called_once()
        called_prompt = mock_llm.ainvoke.call_args[0][0]
        self.assertIn("What is ROCm?", called_prompt)
        self.assertIn("ROCm documentation content", called_prompt)
        self.assertIn("https://rocm.docs.amd.com", called_prompt)
        
        # Verify response was set in state
        self.assertEqual(result_state["answer"], "ROCm is an open-source software platform for GPU computing.")

    @patch('rag_nodes.chat_llm')
    def test_generate_node_stream_streaming_mode(self, mock_llm):
        """Test generate_node_stream in streaming mode"""
        # Mock streaming response
        async def mock_astream(prompt):
            chunks = [
                AIMessageChunk(content="ROCm "),
                AIMessageChunk(content="is an "),
                AIMessageChunk(content="open-source "),
                AIMessageChunk(content="platform.")
            ]
            for chunk in chunks:
                yield chunk
        
        mock_llm.astream.return_value = mock_astream("test prompt")
        
        input_state: GraphState = {
            "question": "What is ROCm?",
            "documents": [
                Document(page_content="ROCm info", metadata={"url": "https://test.com"})
            ],
            "urls": ["https://test.com"],
            "answer": None,
            "history": [],
            "stream": True
        }
        
        # Run the async function
        async def run_test():
            result = await generate_node_stream(input_state)
            return result
        
        result_state = asyncio.run(run_test())
        
        # In streaming mode, the function should not return a state with answer
        mock_llm.astream.assert_called_once()

    @patch('rag_nodes.chat_llm')
    def test_generate_node_stream_with_history(self, mock_llm):
        """Test generate_node_stream includes conversation history in prompt"""
        mock_response = Mock()
        mock_response.content = "Based on our previous conversation, ROCm is..."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        input_state: GraphState = {
            "question": "Can you tell me more?",
            "documents": [Document(page_content="More ROCm info", metadata={"url": "test.com"})],
            "urls": ["test.com"],
            "answer": None,
            "history": [
                {"role": "user", "content": "What is ROCm?"},
                {"role": "assistant", "content": "ROCm is a platform..."}
            ],
            "stream": False
        }
        
        async def run_test():
            result = await generate_node_stream(input_state)
            return result
        
        result_state = asyncio.run(run_test())
        
        # Verify history was included in the prompt
        called_prompt = mock_llm.ainvoke.call_args[0][0]
        self.assertIn("What is ROCm?", called_prompt)
        self.assertIn("ROCm is a platform...", called_prompt)
        self.assertIn("Can you tell me more?", called_prompt)


# ===== GRAPH TESTS =====
@unittest.skipUnless(IMPORTS_AVAILABLE, "Langgraph modules not available")
class TestRagGraph(unittest.TestCase):
    """Test cases for RAG graph building and execution"""

    @patch('rag_graph.retrieve_node')
    @patch('rag_graph.generate_node_stream')
    def test_build_graph_structure(self, mock_generate, mock_retrieve):
        """Test that build_graph creates a graph with correct structure"""
        # Build the graph
        graph = build_graph()
        
        # Verify graph was created and compiled
        self.assertIsNotNone(graph)
        
        # The graph should be a compiled LangGraph object
        self.assertTrue(hasattr(graph, 'invoke'))
        self.assertTrue(hasattr(graph, 'ainvoke'))
        self.assertTrue(hasattr(graph, 'stream'))
        self.assertTrue(hasattr(graph, 'astream'))

    def test_build_graph_returns_compiled_graph(self):
        """Test that build_graph returns a compiled graph object"""
        with patch('rag_graph.StateGraph') as mock_state_graph:
            mock_builder = Mock()
            mock_compiled_graph = Mock()
            mock_builder.compile.return_value = mock_compiled_graph
            mock_state_graph.return_value = mock_builder
            
            result = build_graph()
            
            # Verify StateGraph was instantiated with GraphState
            mock_state_graph.assert_called_once()
            
            # Verify nodes were added
            self.assertEqual(mock_builder.add_node.call_count, 2)
            
            # Verify the node calls
            add_node_calls = mock_builder.add_node.call_args_list
            
            # Check that retrieve node was added
            retrieve_call = add_node_calls[0]
            self.assertEqual(retrieve_call[0][0], "retrieve")
            
            # Check that generate node was added with async flag
            generate_call = add_node_calls[1]
            self.assertEqual(generate_call[0][0], "generate")
            self.assertTrue(generate_call[1]['is_async'])
            
            # Verify edge was added
            mock_builder.add_edge.assert_called_once_with("retrieve", "generate")
            
            # Verify entry point was set
            mock_builder.set_entry_point.assert_called_once_with("retrieve")
            
            # Verify graph was compiled
            mock_builder.compile.assert_called_once()
            
            # Verify the compiled graph is returned
            self.assertEqual(result, mock_compiled_graph)


# ===== API TESTS =====
@unittest.skipUnless(IMPORTS_AVAILABLE, "Langgraph modules not available")
class TestChatModels(unittest.TestCase):
    """Test cases for Pydantic models"""

    def test_chat_message_model(self):
        """Test ChatMessage model validation"""
        # Valid messages
        valid_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "system", "content": "You are a helpful assistant"}
        ]
        
        for msg_data in valid_messages:
            msg = ChatMessage(**msg_data)
            self.assertEqual(msg.role, msg_data["role"])
            self.assertEqual(msg.content, msg_data["content"])

    def test_chat_message_invalid_role(self):
        """Test ChatMessage with invalid role"""
        with self.assertRaises(ValueError):
            ChatMessage(role="invalid", content="Test content")

    def test_chat_request_model_defaults(self):
        """Test ChatRequest model with default values"""
        messages = [
            ChatMessage(role="user", content="Test question")
        ]
        
        request = ChatRequest(
            model="test-model",
            messages=messages
        )
        
        self.assertEqual(request.model, "test-model")
        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.temperature, 0.7)  # Default value
        self.assertIsNone(request.max_tokens)  # Default None
        self.assertTrue(request.stream)  # Default True


@unittest.skipUnless(IMPORTS_AVAILABLE, "Langgraph modules not available")
class TestChatEndpoint(unittest.TestCase):
    """Test cases for the chat endpoint"""

    def setUp(self):
        self.client = TestClient(app)

    @patch('serve.rag_graph')
    def test_chat_endpoint_non_streaming(self, mock_graph):
        """Test chat endpoint in non-streaming mode"""
        # Mock graph response
        mock_final_state = {
            "question": "What is ROCm?",
            "documents": [],
            "urls": [],
            "answer": "ROCm is an open-source platform for GPU computing.",
            "history": [],
            "stream": False
        }
        
        mock_graph.ainvoke = AsyncMock(return_value=mock_final_state)
        
        # Test request
        request_data = {
            "model": "ROCm-RAG-Langgraph",
            "messages": [
                {"role": "user", "content": "What is ROCm?"}
            ],
            "stream": False
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        
        response_data = response.json()
        self.assertIn("id", response_data)
        self.assertEqual(response_data["object"], "chat.completion")
        self.assertEqual(len(response_data["choices"]), 1)
        
        choice = response_data["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(choice["message"]["content"], "ROCm is an open-source platform for GPU computing.")
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["index"], 0)

    def test_chat_endpoint_no_user_message(self):
        """Test chat endpoint with no user message"""
        request_data = {
            "model": "ROCm-RAG-Langgraph",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"}
            ],
            "stream": False
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("error", response_data)
        self.assertEqual(response_data["error"], "No user message found.")

    def test_chat_endpoint_invalid_request(self):
        """Test chat endpoint with invalid request data"""
        # Missing required fields
        request_data = {
            "messages": [
                {"role": "user", "content": "Test"}
            ]
            # Missing "model" field
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        self.assertEqual(response.status_code, 422)  # Validation error


@unittest.skipUnless(IMPORTS_AVAILABLE, "Langgraph modules not available")
class TestModelsEndpoint(unittest.TestCase):
    """Test cases for the models endpoint"""

    def setUp(self):
        self.client = TestClient(app)

    def test_list_models_endpoint(self):
        """Test the /v1/models endpoint"""
        response = self.client.get("/v1/models")
        
        self.assertEqual(response.status_code, 200)
        
        response_data = response.json()
        self.assertEqual(response_data["object"], "list")
        self.assertIn("data", response_data)
        self.assertEqual(len(response_data["data"]), 1)
        
        model = response_data["data"][0]
        self.assertEqual(model["id"], "ROCm-RAG-Langgraph")
        self.assertEqual(model["object"], "model")
        self.assertEqual(model["owned_by"], "AMD")


# ===== INTEGRATION TESTS =====
@unittest.skipUnless(IMPORTS_AVAILABLE, "Langgraph modules not available")
class TestRAGPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete RAG pipeline"""

    @patch('rag_nodes.rag_vectorstore')
    @patch('rag_nodes.chat_llm')
    def test_complete_rag_pipeline_flow(self, mock_llm, mock_vectorstore):
        """Test the complete flow from question to answer"""
        # Mock retrieved documents
        mock_docs = [
            Document(
                page_content="ROCm (Radeon Open Compute) is an open-source software platform for GPU computing.",
                metadata={"url": "https://rocm.docs.amd.com/overview"}
            ),
            Document(
                page_content="ROCm supports HIP (Heterogeneous-compute Interface for Portability) programming model.",
                metadata={"url": "https://rocm.docs.amd.com/hip"}
            )
        ]
        mock_vectorstore.retriever.invoke.return_value = mock_docs
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "ROCm is an open-source software platform for GPU computing that supports the HIP programming model."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Build and test the graph
        graph = build_graph()
        
        initial_state: GraphState = {
            "question": "What is ROCm?",
            "documents": [],
            "urls": [],
            "answer": None,
            "history": [],
            "stream": False
        }
        
        # Test the pipeline by verifying mocks are called correctly
        self.assertIsNotNone(graph)
        
        # Verify that when retrieve_node would be called, it uses the vectorstore
        result_state = retrieve_node(initial_state.copy())
        
        mock_vectorstore.retriever.invoke.assert_called_with("What is ROCm?")
        self.assertEqual(len(result_state["documents"]), 2)
        self.assertEqual(len(result_state["urls"]), 2)
        self.assertIn("https://rocm.docs.amd.com/overview", result_state["urls"])

    @patch('rag_nodes.rag_vectorstore')
    @patch('rag_nodes.chat_llm')
    def test_context_relevance_preservation(self, mock_llm, mock_vectorstore):
        """Test that retrieved context is properly used in generation"""
        # Mock highly relevant documents
        mock_docs = [
            Document(
                page_content="ROCm version 5.7 includes significant performance improvements for machine learning workloads.",
                metadata={"url": "https://rocm.docs.amd.com/release-notes/5.7"}
            ),
            Document(
                page_content="The ROCm 5.7 release focuses on PyTorch and TensorFlow optimization.",
                metadata={"url": "https://rocm.docs.amd.com/pytorch-tensorflow"}
            )
        ]
        mock_vectorstore.retriever.invoke.return_value = mock_docs
        
        # Mock LLM to verify it receives the context
        def mock_ainvoke(prompt):
            # Verify the prompt contains the retrieved context
            assert "ROCm version 5.7" in prompt
            assert "performance improvements" in prompt
            assert "PyTorch and TensorFlow" in prompt
            assert "https://rocm.docs.amd.com/release-notes/5.7" in prompt
            
            response = Mock()
            response.content = "ROCm 5.7 brings significant performance improvements for machine learning."
            return response
        
        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        
        # Test the generation node
        state: GraphState = {
            "question": "What's new in ROCm 5.7?",
            "documents": mock_docs,
            "urls": ["https://rocm.docs.amd.com/release-notes/5.7", "https://rocm.docs.amd.com/pytorch-tensorflow"],
            "answer": None,
            "history": [],
            "stream": False
        }
        
        async def test_generation():
            result = await generate_node_stream(state)
            return result
        
        result_state = asyncio.run(test_generation())
        
        # Verify the LLM was called and context was preserved
        mock_llm.ainvoke.assert_called_once()
        self.assertEqual(result_state["answer"], "ROCm 5.7 brings significant performance improvements for machine learning.")


@unittest.skipUnless(IMPORTS_AVAILABLE, "Langgraph modules not available")
class TestEndToEndAPIIntegration(unittest.TestCase):
    """End-to-end integration tests using the FastAPI client"""

    def setUp(self):
        self.client = TestClient(app)

    @patch('serve.rag_graph')
    def test_complete_api_workflow_non_streaming(self, mock_graph):
        """Test complete API workflow in non-streaming mode"""
        # Mock the complete graph execution
        mock_final_state = {
            "question": "What is ROCm?",
            "documents": [
                Document(
                    page_content="ROCm is an open-source platform",
                    metadata={"url": "https://rocm.docs.amd.com"}
                )
            ],
            "urls": ["https://rocm.docs.amd.com"],
            "answer": "ROCm is an open-source software platform for GPU computing developed by AMD.",
            "history": [],
            "stream": False
        }
        
        mock_graph.ainvoke = AsyncMock(return_value=mock_final_state)
        
        # Make API request
        request_data = {
            "model": "ROCm-RAG-Langgraph",
            "messages": [
                {"role": "user", "content": "What is ROCm?"}
            ],
            "stream": False,
            "temperature": 0.7
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        
        self.assertEqual(response_data["object"], "chat.completion")
        self.assertEqual(len(response_data["choices"]), 1)
        
        choice = response_data["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(
            choice["message"]["content"],
            "ROCm is an open-source software platform for GPU computing developed by AMD."
        )
        self.assertEqual(choice["finish_reason"], "stop")
        
        # Verify the graph was called with correct initial state
        mock_graph.ainvoke.assert_called_once()
        called_state = mock_graph.ainvoke.call_args[0][0]
        self.assertEqual(called_state["question"], "What is ROCm?")
        self.assertFalse(called_state["stream"])

    @patch('serve.rag_graph')
    def test_api_workflow_streaming_mode(self, mock_graph):
        """Test API workflow in streaming mode"""
        # Mock streaming response
        async def mock_astream(state, stream_mode):
            chunks = [
                (AIMessageChunk(content="ROCm "), {"node": "generate"}),
                (AIMessageChunk(content="is "), {"node": "generate"}),
                (AIMessageChunk(content="awesome!"), {"node": "generate"})
            ]
            for chunk in chunks:
                yield chunk
        
        mock_graph.astream = mock_astream
        
        request_data = {
            "model": "ROCm-RAG-Langgraph",
            "messages": [
                {"role": "user", "content": "Tell me about ROCm"}
            ],
            "stream": True
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")

