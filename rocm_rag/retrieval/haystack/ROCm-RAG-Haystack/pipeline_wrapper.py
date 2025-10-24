"""hayhooks wrapper for haystack query pipeline."""

import os
import re
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from haystack.components.builders import PromptBuilder, ChatPromptBuilder
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import ToolsType
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

from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable
from haystack.utils.http_client import init_http_client


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

class ChatGeneratorWrapper(OpenAIChatGenerator):
    """
    A wrapper around OpenAIChatGenerator that processes citations.
    
    This wrapper inherits from OpenAIChatGenerator and processes the generated text to:
    1. Convert [citation:i] format to [i] format
    2. Renumber citations sequentially [1], [2], [3], etc.
    3. Update the References section to match the renumbered citations
    """
    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        model,
        api_base_url
    ):
        super().__init__(model=model, 
                         api_base_url=api_base_url)

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        *,
        tools: Optional[ToolsType] = None,
        tools_strict: Optional[bool] = None,
    ):
        """
        Run the OpenAI chat generator and process citations in the response.
        """
        # Call the parent class method to get the original response
        result = super().run(messages=messages)
        
        if not result or "replies" not in result or not result["replies"]:
            return result
            
        # Process each reply to fix citations
        processed_replies = []
        for reply in result["replies"]:
            processed_content = self._process_citations(reply.content)
            # Create a new ChatMessage with the processed content
            processed_reply = ChatMessage.from_assistant(processed_content)
            processed_replies.append(processed_reply)
        
        # Return the result with processed replies
        result["replies"] = processed_replies
        return result
    
    def _process_citations(self, text: str) -> str:
        """
        Process citations in the text to:
        1. Convert [citation:i] to [i]
        2. Renumber citations sequentially 
        3. Update References section accordingly
        """
        if not text:
            return text
            
        # Step 1: Find all [citation:X] patterns and extract the original numbers
        citation_pattern = r'\[citation:(\d+)\]'
        citations_found = re.findall(citation_pattern, text)
        
        if not citations_found:
            return text
            
        # Step 2: Create mapping from original numbers to sequential numbers
        unique_citations = []
        for citation in citations_found:
            if citation not in unique_citations:
                unique_citations.append(citation)
        
        # Create mapping: original_number -> new_sequential_number
        citation_mapping = {}
        for i, original_num in enumerate(unique_citations, 1):
            citation_mapping[original_num] = str(i)
        
        # Step 3: Replace [citation:X] with [new_number] in the text
        def replace_citation(match):
            original_num = match.group(1)
            new_num = citation_mapping[original_num]
            return f'[{new_num}]'
        
        processed_text = re.sub(citation_pattern, replace_citation, text)
        
        # Step 4: Update References section if it exists
        processed_text = self._update_references_section(processed_text, citation_mapping)
        
        return processed_text
    
    def _update_references_section(self, text: str, citation_mapping: dict) -> str:
        """
        Update the References section to match the renumbered citations.
        """
        # Look for References section (case-insensitive)
        references_pattern = r'(References?:?\s*\n)(.*?)(?=\n\n|\Z)'
        references_match = re.search(references_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if not references_match:
            return text
            
        references_header = references_match.group(1)
        references_content = references_match.group(2)
        
        # Find all reference entries in format [X] - URL or similar
        reference_pattern = r'\[(\d+)\]\s*-\s*(.+?)(?=\n\[|\Z)'
        reference_matches = re.findall(reference_pattern, references_content, re.DOTALL)
        
        if not reference_matches:
            return text
            
        # Create new references section with updated numbering
        new_references = []
        
        # Create reverse mapping to find which original number maps to each new number
        reverse_mapping = {v: k for k, v in citation_mapping.items()}
        
        # Build new references in sequential order
        for new_num in sorted(reverse_mapping.keys(), key=int):
            original_num = reverse_mapping[new_num]
            
            # Find the reference content for this original number
            for ref_num, ref_content in reference_matches:
                if ref_num == original_num:
                    new_references.append(f'[{new_num}] - {ref_content.strip()}')
                    break
        
        # Replace the references section
        new_references_section = references_header + '\n'.join(new_references)
        
        # Replace the original references section
        updated_text = re.sub(
            references_pattern, 
            new_references_section, 
            text, 
            flags=re.IGNORECASE | re.DOTALL
        )
        
        return updated_text


class PipelineWrapper(BasePipelineWrapper):
    """Haystack query pipeline wrapper for hayhooks."""


    def _parse_generation_params(self, body: dict) -> dict:
        """
        Parse OpenAI-compatible generation parameters from request body.
        Only includes parameters that are explicitly provided in the request.

        Args:
            body: Request body dictionary containing user parameters

        Returns:
            Dictionary of generation parameters to pass to the LLM
        """
        generation_params = {}

        # Common OpenAI parameters that can be overridden
        param_mapping = {
            'temperature': 'temperature',
            'max_tokens': 'max_tokens',
            'top_p': 'top_p',
            'frequency_penalty': 'frequency_penalty',
            'presence_penalty': 'presence_penalty',
            'stop': 'stop',
            'n': 'n'
        }

        # Extract parameters from body if they exist and are not None
        for body_key, param_key in param_mapping.items():
            if body_key in body and body[body_key] is not None:
                generation_params[param_key] = body[body_key]

        return generation_params

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
        # retriever = LoggingWeaviateEmbeddingRetriever(document_store=document_store)
        retriever = WeaviateEmbeddingRetriever(document_store=document_store,
                                               top_k=config.ROCM_RAG_HAYSTACK_TOP_K_RANKING,
                                               certainty=config.ROCM_RAG_HAYSTACK_CERTAINTY_THRESHOLD)

        # Define the prompt template, with retrieved context and query
        # template = """
        # Given the following information, answer the question in detail.

        # Context:
        # {% for document in documents %}
        #     {{ document.content }}
        # {% endfor %}

        # {% if urls|length != 0 %}
        #     Reference Links:
        #     {% for url in urls %}
        #         - {{ url }}
        #     {% endfor %}
        #     Append reference links to end of answer.
        # {% endif %}

        # History:
        # {% for message in query[:-1] %}
        #     {{ message['role'] }}: {{ message['content'] }}
        # {% endfor %}

        # Question: {{ query[-1] }}?
        # """

        # # Initialize prompt builder
        # prompt_builder = PromptBuilderWithRefLinks(template=template)

        # Initialize OpenAI generator
        # generator = OpenAIGenerator(
        #     model=llm_model,
        #     api_base_url = llm_api_base_url
        # )

        template = [
            ChatMessage.from_user(
                """
                # The following are the list of documents related to the user's question:
                {% for document in documents %}
                    [document {{ loop.index }} begin page_url {{ document.meta.url }}] {{ document.content }} [document {{ loop.index }} end] \n
                {% endfor %}

                In the documents list I provide to you, each document is formatted as [document X begin page_url Z]...[document X end], where X represents the numerical index of each document and Z is the url of the document. Please cite the context at the end of the relevant sentence when appropriate. Use the citation format [citation:X] in the corresponding part of your answer. If a sentence is derived from multiple contexts, list all relevant citation numbers, such as [citation:3][citation:5]. Be sure not to cluster all citations at the end; instead, include them in the corresponding parts of the answer.
                At the end of the answer, add a section called 'References' and list all the citations with the format '[X] - Z' where X is the index of the document and Z is the url of the document cited.
                When responding, please keep the following points in mind:
                - Not all content in the documents list is closely related to the user's question. You need to evaluate and filter the documents based on the question.
                - For listing-type questions (e.g., listing all ROCm features), try to limit the answer to 10 key points and inform the user that they can refer to the document sources for complete information. Prioritize providing the most complete and relevant items in the list. Avoid mentioning content not provided in the documents unless necessary.
                - For creative tasks (e.g., writing an essay), ensure that references are cited within the body of the text, such as [citation:3][citation:5], rather than only at the end of the text. You need to interpret and summarize the user's requirements, choose an appropriate format, fully utilize the documents listed, extract key information, and generate an answer that is insightful, creative, and professional. Extend the length of your response as much as possible, addressing each point in detail and from multiple perspectives, ensuring the content is rich and thorough.
                - If the response is lengthy, structure it well and summarize it in paragraphs. If a point-by-point format is needed, try to limit it to 5 points and merge related content.
                - For objective Q&A, if the answer is very brief, you may add one or two related sentences to enrich the content.
                - Choose an appropriate and visually appealing format for your response based on the user's requirements and the content of the answer, ensuring strong readability.
                - Your answer should synthesize information from multiple relevant documents and avoid repeatedly citing the same document.
                - Unless the user requests otherwise, your response should be in the same language as the user's question.
                - If the user's question is not related to the documents list, you should respond with "I'm sorry, I don't have information about that."

                # The user's chat history is:
                {% for message in query[:-1] %}
                    {{ message['role'] }}: {{ message['content'] }}
                {% endfor %}

                # The user's message is:
                Question: {{ query[-1] }}?
                """
            )
        ]

        prompt_builder = ChatPromptBuilder(template=template)

        chat_generator = ChatGeneratorWrapper(
            model=llm_model,
            api_base_url=llm_api_base_url
        )

        query_pipeline = Pipeline()
        query_pipeline.add_component("text_embedder", text_embedder)
        query_pipeline.add_component("retriever", retriever)
        query_pipeline.add_component("prompt_builder", prompt_builder)
        query_pipeline.add_component("llm", chat_generator)

        query_pipeline.connect("text_embedder.embedding", "retriever")
        query_pipeline.connect("retriever", "prompt_builder.documents")
        query_pipeline.connect("prompt_builder.prompt", "llm.messages")
        self.pipeline = query_pipeline

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
        last_query = get_last_user_message(messages)

        # Parse generation parameters from user request
        generation_params = self._parse_generation_params(body)

        run_args = {
            "text_embedder": {"text": last_query},
            "prompt_builder": {"query": messages},
            "llm": {"generation_kwargs": generation_params}
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

