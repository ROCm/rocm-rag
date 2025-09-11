from langgraph.graph import StateGraph
from rag_nodes import retrieve_node, generate_node_stream
from rag_types import GraphState

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node_stream, is_async=True)
    builder.add_edge("retrieve", "generate")
    builder.set_entry_point("retrieve")

    return builder.compile()