from typing import TypedDict, Any
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from app.services.vectordb_service import search


# Define the LLM
llm = ChatOllama(
    model="mistral",
    temperature=0.2
)

class SearchTextState(TypedDict, total=False):
    query: str
    k: int
    docs: list[dict[str, Any]]
    answer: str

class NERSearchState(TypedDict, total=False):
    query: str
    k: int
    passages: list
    combined_text: str
    entities: dict
    answer: str

# =========NODE DEFINITIONS=========

def retrieve_freewriting_node(state: SearchTextState) -> SearchTextState:
    """Retrieve documents from the freewriting collection"""
    query = state.get("query", "")
    k = state.get("k", 3)

    results = search(query, k=k, collection="freewriting")
    return {"docs": results}

def generate_answer_node(state: SearchTextState) -> SearchTextState:
    """Generate answer based on retrieved documents"""
    query = state.get("query", "")
    docs = state.get("docs", [])

    # Combine document texts
    combined_docs = "\n\n".join(passage["text"] for passage in docs) if docs else "No relevant information found."

    prompt = (
        f"You are a writing assistant named Walt Bot."
        f"You are very helpful and you offer information that assists the writer who is speaking with you. "
        f"You don't do writing for them unless they specifically ask you, but you provide information that helps guide them to do it themselves. "
        f"You speak in a poetic, but very accurate and concise way. "
        f"Your style of writing is reminiscent of Walt Whitman, Lon Milo DuQuette, and Carl Sagan."
        f"Answer the User's Query based on the Extracted Data below. "
        f"If there's no relevant information stored, you can say that.\n\n"
        f"Extracted Data:\n{combined_docs}\n\n"
        f"User Query: {query}\n\n"
        f"Answer: "
    )

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)

    return {"answer": answer}

# NER node definitions
def retrieve_passages_node(state: NERSearchState) -> NERSearchState:
    """Retrieve passages from freewriting collection"""
    query = state.get("query", "")
    k = state.get("k", 3)
    result = search(query, k=k, collection="freewriting")
    return {"passages": result}

def combine_text_node(state: NERSearchState) -> NERSearchState:
    """Combine passage texts"""
    passages = state.get("passages", [])
    combined_text = "\n\n".join(passage["text"] for passage in passages)
    return {"combined_text": combined_text}

def extract_entities_node(state: NERSearchState) -> NERSearchState:
    """Extract named entities from combined text"""
    from app.services.vectordb_service import extract_entities
    combined_text = state.get("combined_text", "")
    entities = extract_entities(combined_text)
    return {"entities": entities}

def generate_ner_answer_node(state: NERSearchState) -> NERSearchState:
    """Generate answer based on extracted entities"""
    query = state.get("query", "")
    entities = state.get("entities", {})

    prompt = (
        f"Based on the following extracted entities from the freewriting sample, "
        f"{entities}\n"
        f"Answer the User's NER-based query with ONLY the data you see here.\n"
        f"User query: {query}"
    )

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)

    return {"answer": answer}

# ==========BUILD GRAPH==========

def build_search_text_graph():
    """Build the LangGraph workflow for freewriting search"""
    workflow = StateGraph(SearchTextState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_freewriting_node)
    workflow.add_node("generate", generate_answer_node)

    # Define flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile with memory (optional, to track state across calls)
    return workflow.compile(checkpointer=MemorySaver())

# Create singleton instance
search_text_graph = build_search_text_graph()

# Build NER graph
def build_ner_search_graph():
    """Build the LangGraph workflow for NER-based search"""
    workflow = StateGraph(NERSearchState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_passages_node)
    workflow.add_node("combine", combine_text_node)
    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("generate", generate_ner_answer_node)

    # Define flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "combine")
    workflow.add_edge("combine", "extract_entities")
    workflow.add_edge("extract_entities", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile(checkpointer=MemorySaver())

# Create singleton instance
ner_search_graph = build_ner_search_graph()