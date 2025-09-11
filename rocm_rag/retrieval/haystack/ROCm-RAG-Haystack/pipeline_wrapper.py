"""hayhooks wrapper for haystack query pipeline."""

import os
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack import Pipeline
from haystack.components.embedders import (
    OpenAITextEmbedder
)
from haystack_integrations.document_stores.weaviate.document_store import (
    WeaviateDocumentStore,
)

from typing import Any, Dict, List, Optional, Union

from haystack import Document, component


from hayhooks.server.pipelines.utils import get_last_user_message
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks import streaming_generator
from typing import Generator, AsyncGenerator, List, Union
from rocm_rag import config


class LoggingWeaviateEmbeddingRetriever(WeaviateEmbeddingRetriever):
    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        distance: Optional[float] = None,
        certainty: Optional[float] = None
    ) -> Dict[str, List[Document]]:
        """
        Run the retriever with the given query embedding and optional filters.
        Logs the details of each retrieved document.
        """
        documents = super().run(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k if top_k is not None else config.ROCM_RAG_HAYSTACK_TOP_K_RANKING,
            distance=distance,
            certainty=certainty if certainty is not None else config.ROCM_RAG_HAYSTACK_CERTAINTY_THRESHOLD
        )["documents"]

        return {"documents": documents}

class PromptBuilderWithRefLinks(PromptBuilder):
    @component.output_types(prompt=str)
    def run(self, template: Optional[str] = None, template_variables: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Run the prompt builder with the given query and documents.
        Adds reference links to the prompt.
        """
        # Generate the base prompt using the parent class method
        urls = list(set([x.to_dict()["url"] for x in kwargs.get("documents", [])]))
        kwargs["urls"] = urls  # Add URLs to the template variables
        result = super().run(template=template, template_variables=template_variables, **kwargs)
        prompt = result.get("prompt", "")
        return {"prompt": prompt}

class PipelineWrapper(BasePipelineWrapper):
    """Haystack query pipeline wrapper for hayhooks."""

    def setup(self) -> None:
        weaviate_url = f"{config.ROCM_RAG_WEAVIATE_URL}:{config.ROCM_RAG_WEAVIATE_PORT}"
        embedder_api_base_url = f"{config.ROCM_RAG_EMBEDDER_API_BASE_URL}:{config.ROCM_RAG_EMBEDDER_API_PORT}/v1"
        embedder_model = config.ROCM_RAG_EMBEDDER_MODEL
        llm_api_base_url = f"{config.ROCM_RAG_LLM_API_BASE_URL}:{config.ROCM_RAG_LLM_API_PORT}/v1"
        llm_model = config.ROCM_RAG_LLM_MODEL

        # Initialize WeaviateDocumentStore
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
        document_store = WeaviateDocumentStore(url=weaviate_url, collection_settings=collection_settings)

        # Initialize text embedder
        text_embedder = OpenAITextEmbedder(
                api_base_url=embedder_api_base_url, 
                model=embedder_model
            )
        
        # Initialize retriever
        retriever = LoggingWeaviateEmbeddingRetriever(document_store=document_store)

        # Define the prompt template, with retrieved context and query
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

        # Initialize prompt builder
        prompt_builder = PromptBuilderWithRefLinks(template=template)

        # Initialize OpenAI generator
        generator = OpenAIGenerator(
            model=llm_model,
            api_base_url = llm_api_base_url
        )

        query_pipeline = Pipeline()
        query_pipeline.add_component("text_embedder", text_embedder)
        query_pipeline.add_component("retriever", retriever)
        query_pipeline.add_component("prompt_builder", prompt_builder)
        query_pipeline.add_component("llm", generator)
        query_pipeline.connect("text_embedder.embedding", "retriever")
        query_pipeline.connect("retriever", "prompt_builder.documents")
        query_pipeline.connect("prompt_builder", "llm")
        self.pipeline = query_pipeline
    
    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
        last_query = get_last_user_message(messages)
        run_args = {
            "text_embedder": {"text": last_query},
            "prompt_builder": {"query": messages},
        }

        # If request has stream=True, return a generator
        if "stream" in body and body["stream"]:
            return streaming_generator(
                pipeline=self.pipeline,
                pipeline_run_args=run_args,
            )
        else:
            # Non-streaming: run the pipeline once and return full text
            result = self.pipeline.run(run_args)
            replies = result.get("llm", {}).get("replies", [])
            if not replies:
                return ""
            return replies[0]

