from typing import TypedDict, Any, Annotated

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, add_messages

from app.services.vectordb_service import search

# define the LLM
llm = ChatOllama(
    model="mistral",
    temperature=0.2
)

class GraphState(TypedDict, total=False):

    query: str

    route: str

    docs: list[dict[str, Any]]

    answer: str

    message_memory: Annotated[list[BaseMessage], add_messages]

# =========NODE DEFINITIONS=========

# Each node takes in graph state and returns graph state

# Route Node

def route_node(state: GraphState) -> GraphState:

    query = state.get("query", "").lower()


    if any(word in query for word in ["passage", "passages", "archive", "archives", "history"]):
        return {"route":"passages"} # this return adds the route to State

    if any(word in query for word in ["freewrite", "freewriting"]):
        return {"route":"freewriting"}

    return {"route":"chat"}

def extract_passages_node(state: GraphState) -> GraphState:

    query = state.get("query", "")
    results = search(query, k=5, collection="passages")

    return {"docs":results}


def extract_text_node(state: GraphState) -> GraphState:

    query = state.get("query", "")
    results = search(query, k=10, collection="freewriting")
    return {"docs":results}

def answer_with_context_node(state: GraphState) -> GraphState:

    query = state.get("query", "")
    docs = state.get("docs", [])
    combined_docs = "\n\n".join(passage["text"] for passage in docs)

    prompt = (
        f"You are a writing assistant."
        f"You are very helpful and you offer information that assists the writer who is speaking with you."
        f"You don't do writing for them unless they specifically ask you, but you provide information that helps guide them to do it themselves."
        f"You speak in a poetic, but very accurate and concise way."
        f"Your style of writing is vaguely reminiscent of Walt Whitman and Carl Sagan."
        f"Answer the User's Query based ONLY on the Extracted Data below."
        f"If the data doesn't help, say you do not know."
        f"Extracted Data:\n{combined_docs}"
        f"User Query:\n{query}"
        f"Answer: "
    )

    response = llm.invoke(prompt)

    return {"answer":response}

def general_chat_node(state: GraphState) -> GraphState:

    prompt = (
        f"""You are a writing assistant.
        You are very helpful and you offer information that assists the writer who is speaking with you.
        You don't do writing for them unless they specifically ask you, but you provide information that helps guide them to do it themselves.
        You speak in a poetic, but very accurate and concise way.
        Your style of writing is vaguely reminiscent of Walt Whitman and Carl Sagan.
        You have context from previous interactions: \n{state.get('message_memory')}
        Answer the User's Query to the best of your ability.
        User Query:\n{state.get('query','')}
        Answer: """
    )

    result = llm.invoke(prompt).content

    return {"answer":result,
            "message_memory": [
                HumanMessage(content=state.get("query")),
                AIMessage(content=result)
            ]
            }

# ==========END OF NODE DEFINITIONS==========

def build_graph():

    build = StateGraph(GraphState)

    build.add_node("route", route_node)
    build.add_node("extract_passages", extract_passages_node)
    build.add_node("extract_text", extract_text_node)
    build.add_node("answer_with_context_node", answer_with_context_node)
    build.add_node("general_chat_node", general_chat_node)

    build.set_entry_point("route")

    build.add_conditional_edges(
        "route",
        lambda state: state["route"],

        {
            "passages":"extract_passages",
            "freewriting":"extract_text",
            "chat":"general_chat_node"
        }
    )

    build.add_edge("extract_passages", "answer_with_context_node")
    build.add_edge("extract_text", "answer_with_context_node")

    build.set_finish_point("answer_with_context_node")
    build.set_finish_point("general_chat_node")

    return build.compile(checkpointer=MemorySaver())

# make a single graph instance (singleton) - ensure only one instance of the graph exists
langgraph = build_graph()