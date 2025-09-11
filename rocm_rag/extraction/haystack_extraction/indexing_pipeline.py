# This script uses Haystack to index documents into a Weaviate vector database.
# It reads text files from a specified directory, split the content, and converts them into Document objects,
# then embeds them using specified embedder model, and writes them to the Weaviate document store.

import os
from tqdm import tqdm
from haystack import Pipeline
from haystack import Document
from haystack import component
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentPreprocessor
from haystack.components.embedders import (
    OpenAIDocumentEmbedder
)
from haystack_integrations.document_stores.weaviate.document_store import (
    WeaviateDocumentStore,
)
from typing import List
import torch
from torch.nn.functional import cosine_similarity
from rocm_rag import config


@component
class SemanticChunkMerger:
    def __init__(self, similarity_threshold, max_chunk_length, embedder: OpenAIDocumentEmbedder = None):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_length = max_chunk_length
        self.embedder = embedder

    @component.output_types(documents=List[Document])
    def run(self, documents: list[Document], **kwargs):
        if not documents:
            return {"documents": []}
        
        url = documents[0].meta.get("url", "unknown_url")
        domain = documents[0].meta.get("domain", "unknown_domain")
        if not documents:
            print(f"No documents to process for URL: {url}")
        embedded_chunks = self.embedder.run(documents=documents)["documents"]
        print("Number of chunks before merging:", len(embedded_chunks))
            
        
        merged_documents = []
        current_chunk = embedded_chunks[0].content
        current_embedding = torch.tensor(embedded_chunks[0].embedding)

        if len(embedded_chunks) == 1:
            merged_documents.append(Document(
                content=current_chunk,
                embedding=current_embedding.tolist(),
                meta={"url": url, "domain": domain}
            ))
            return {"documents": merged_documents}
        
        current_chunk = embedded_chunks[0].content
        current_embedding = torch.tensor(embedded_chunks[0].embedding)

        # The last document from embedder is not processed, so it does not have an embedding
        for i in tqdm(range(1, len(embedded_chunks))):
            next_chunk = embedded_chunks[i].content
            next_embedding = torch.tensor(embedded_chunks[i].embedding)
            # stops if merging chunks would exceed the max length
            if len(current_chunk) + len(next_chunk) > self.max_chunk_length:
                merged_documents.append(Document(
                    content=current_chunk,
                    embedding=current_embedding.tolist(),
                    meta={"url": url, "domain": domain}
                ))
                current_chunk = next_chunk
                current_embedding = next_embedding
                continue

            sim = cosine_similarity(current_embedding.unsqueeze(0), next_embedding.unsqueeze(0)).item()

            # merge if similarity is above the threshold
            if sim >= self.similarity_threshold:                
                current_chunk = " ".join([current_chunk, next_chunk])
                current_embedding = torch.tensor(
                    self.embedder.run(documents=[Document(content=current_chunk), Document(content="")])["documents"][0].embedding
                )
            else:
                merged_documents.append(Document(
                    content=current_chunk,
                    embedding=current_embedding.tolist(),
                    meta={"url": url, "domain": domain}
                ))
                current_chunk = next_chunk
                current_embedding = next_embedding

        merged_documents.append(Document(
            content=current_chunk,
            embedding=current_embedding.tolist(),
            meta={"url": url, "domain": domain}
        ))

        return {"documents": merged_documents}


class IndexingPipeline:
    def __init__(self):
        weaviate_url=f"{config.ROCM_RAG_WEAVIATE_URL}:{config.ROCM_RAG_WEAVIATE_PORT}"
        embedder_api_base_url=f"{config.ROCM_RAG_EMBEDDER_API_BASE_URL}:{config.ROCM_RAG_EMBEDDER_API_PORT}/v1"
        embedder_model=config.ROCM_RAG_EMBEDDER_MODEL
        max_chunk_length=config.ROCM_RAG_MAX_CHUNK_LENGTH
        similarity_threshold=config.ROCM_RAG_SIMILARITY_THRESHOLD

        collection_settings = {
            "class": config.ROCM_RAG_WEAVIATE_CLASSNAME,
            "invertedIndexConfig": {"indexNullState": True},
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "url", "dataType": ["string"]},
                {"name": "domain", "dataType": ["string"]},
            ],
            "vectorizer": "none"
        }
        self.document_store = WeaviateDocumentStore(url=weaviate_url, collection_settings=collection_settings)
        self.embedder = OpenAIDocumentEmbedder(
            api_base_url=embedder_api_base_url,
            model=embedder_model
        )
        self.document_preprocessor = DocumentPreprocessor(
            split_by="sentence",
            split_length=2
        )

        self.semantic_chunk_merger = SemanticChunkMerger(
            similarity_threshold=similarity_threshold,
            max_chunk_length=max_chunk_length,
            embedder=self.embedder
        )

        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("preprocessor", self.document_preprocessor)
        self.indexing_pipeline.add_component("semantic_chunk_merger", self.semantic_chunk_merger)
        self.indexing_pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))
        self.indexing_pipeline.connect("preprocessor", "semantic_chunk_merger")
        self.indexing_pipeline.connect("semantic_chunk_merger", "writer")


    def run(self, doc: Document):
        # single page at a time
        results = self.indexing_pipeline.run({"preprocessor": {"documents": [doc]}})
        return results

    def delete_by_url(self, url: str, domain: str):
        filters = {"operator" : "AND",
                    "conditions": [
                        {"field": "url", "value": url, "operator": "=="},
                        {"field": "domain", "value": domain, "operator": "=="}
                    ]}
        docs = self.document_store.filter_documents(filters=filters)
        print(f"Found {len(docs)} documents to delete for {url} in domain {domain}")
        if (docs):
            self.document_store.delete_documents([doc.id for doc in docs])
    
    def insert_page(self, url: str, domain: str, content: str):
        """Helper method to create a Document and run the indexing pipeline."""
        doc = Document(content=content, meta={"url": url, "domain": domain})
        self.run(doc)


