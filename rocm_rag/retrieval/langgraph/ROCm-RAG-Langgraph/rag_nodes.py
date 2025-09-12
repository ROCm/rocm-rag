from rocm_rag.utils.langgraph_vectorstore import RAGVectorStore
from rag_types import GraphState
from langchain_openai import ChatOpenAI
from rocm_rag import config
from langchain_core.messages import AIMessageChunk

llm_api_base_url = f"{config.ROCM_RAG_LLM_API_BASE_URL}:{config.ROCM_RAG_LLM_API_PORT}/v1"
llm_model = config.ROCM_RAG_LLM_MODEL
rag_vectorstore = RAGVectorStore()

# LLM
chat_llm = ChatOpenAI(
    model=llm_model,
    base_url=llm_api_base_url
)

def retrieve_node(state: GraphState) -> GraphState:
    documents = rag_vectorstore.retriever.invoke(state["question"])
    state["documents"] = documents
    state["urls"] = list(set([doc.metadata.get("url", "") for doc in documents if doc.metadata.get("url")]))
    return state

async def generate_node_stream(state: GraphState):
    docs = state["documents"]
    context = "\n\n".join([doc.page_content for doc in docs])
    question = state["question"]
    prompt = f"""Given the following information, answer the question.
    Context: {context}
    Reference Links: {state['urls']}
    Append reference links to the end of the answer.
    Question: {question}
    History: {state['history']}
    """

    # Update chat_llm with request parameters
    generation_kwargs = state.get("generation_kwargs", {})
    
    if state.get("stream", True):
        async for chunk in chat_llm.astream(prompt, **generation_kwargs):            
            yield chunk
    else:
        response = await chat_llm.ainvoke(prompt, **generation_kwargs)
        state["answer"] = response.content
