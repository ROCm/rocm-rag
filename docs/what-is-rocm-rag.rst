.. meta::
  :description: Learn about ROCm-RAG
  :keywords: ROCm, RAG, documentation

.. _what-is-rocm-rag:

*****************
What is ROCm-RAG?
*****************

ROCm-RAG is an optimized Retrieval-Augmented Generation (RAG) implementation designed to run efficiently on AMD GPUs using the ROCm platform. It provides a complete, production-ready solution for building AI applications that combine the power of large language models with real-time information retrieval.

RAG enhances the accuracy and reliability of Large Language Models by exposing them to up-to-date, relevant information.
When a query is received, RAG retrieves relevant documents or information from its knowledge base and then uses this retrieved context, along with the query, to generate accurate and informed responses.
This approach helps reduce hallucinations (making up information) common in standard LLMs, while also enabling the model to access current information not present in its original training data.

Organizations rely on the RAG pipelines (end-to-end systems) that process and manage information from raw data to the final response generation.
These pipelines operate in two main phases, the extraction phase and the retrieval phase: 

- During extraction, documents are processed, split into chunks, converted into vector embeddings (numerical representations of text), and stored in a `Weaviate <https://docs.weaviate.io/weaviate>`__ vector database. 
- In the retrieval phase, when a user asks a question, the pipeline retrieves relevant information and generates a response using an LLM, ensuring that the benefits of RAG are translated into practical, reliable results.

RAG is particularly valuable in enterprise applications where accuracy and verifiable information are crucial, such as customer support systems, research assistants, and documentation tools.

Features and use cases
====================================================================

ROCm-RAG on AMD hardware provides the following key features:

* **Document processing pipeline**: Automated ingestion, chunking, and preprocessing of various document formats
* **Vector embedding generation**: High-performance embedding creation optimized for AMD GPUs
* **Weaviate integration**: Seamless integration with Weaviate vector database for efficient similarity search
* **LLM inference**: Accelerated language model inference using ROCm-optimized frameworks
* **Retrieval optimization**: Advanced retrieval strategies, including semantic search and hybrid search
* **Context management**: Intelligent context window management for optimal LLM performance
* **Multi-document support**: Process and query across multiple document sources simultaneously
* **Customizable chunking**: Flexible text splitting strategies for different document types
* **Query optimization**: Efficient query processing and result ranking
* **API interface**: RESTful API for easy integration into existing applications

ROCm-RAG on AMD hardware includes performance-enhancing features:

* **GPU-accelerated embeddings**: Fast vector embedding generation using AMD GPU compute
* **Optimized inference**: ROCm-tuned LLM inference for reduced latency
* **Memory efficiency**: Intelligent memory management for handling large models and datasets
* **Batch processing**: Efficient batch embedding and inference for high throughput
* **Multi-GPU support**: Scale across multiple AMD GPUs for enterprise workloads

Why ROCm-RAG?
====================================================================

ROCm-RAG is commonly used in the following scenarios:

* **Enterprise knowledge bases**: Build intelligent search systems over internal documentation and knowledge repositories
* **Customer support chatbots**: Create AI assistants that provide accurate answers based on product documentation and support articles
* **Research assistants**: Enable researchers to query and synthesize information from large document collections
* **Legal document analysis**: Search and analyze legal documents with context-aware responses
* **Technical documentation tools**: Build interactive documentation systems that answer user questions accurately
* **Compliance and regulatory systems**: Query regulatory documents and compliance materials with verified information
* **Medical information retrieval**: Access medical literature and clinical guidelines with accurate, sourced responses
* **Financial analysis**: Retrieve and analyze financial reports, market data, and research documents
* **Educational platforms**: Create learning assistants that provide accurate information from course materials
* **Content management systems**: Enhance CMS platforms with intelligent search and question-answering capabilities

