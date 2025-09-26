.. meta::
  :description: Learn about ROCm-RAG
  :keywords: ROCm, RAG, documentation

*****************
What is ROCm-RAG?
*****************

RAG enhances the accuracy and reliability of Large Language Models by exposing it to up-to-date, relevant information.
When a query is received, RAG retrieves relevant documents or information from its knowledge base, then uses this retrieved context alongside the query to generate accurate and informed responses.
This approach helps reduce hallucinations (making up information) common in standard LLMs, while also enabling the model to access current information not present in its original training data.

Organizations rely on the RAG pipelines (end-to-end systems) that process and manage information from raw data to the final response generation.
These pipelines operate in two main phases, the extraction phase and the retrieval phase: 

- During extraction, documents are processed, split into chunks, converted into vector embeddings (numerical representations of text), and stored in a `Weaviate <https://docs.weaviate.io/weaviate>`__ vector database. 
- In the retrieval phase, when a user asks a question, the pipeline retrieves relevant information and generates a response using an LLM, ensuring that the benefits of RAG are translated into practical, reliable results.

RAG is particularly valuable in enterprise applications where accuracy and verifiable information are crucial, such as customer support systems, research assistants, and documentation tools.


