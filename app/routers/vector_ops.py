from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.vectordb_service import ingest_json_service, search, ingest_text, extract_entities
from app.services.vector_langgraph_service import search_text_graph  # New import

router = APIRouter(
    prefix="/vector-ops",
    tags=["vector-ops"],
)

# Pydantic model for document ingestion
class IngestJson(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] | None = None

# model for ingesting raw text
class IngestTextRequest(BaseModel):
    text: str

# model for similarity search request results
class SearchRequest(BaseModel):
    query: str = ""
    k: int = 3

# Endpoint for data ingestion
@router.post("/ingest-json")
async def ingest_json_endpoint(passages: list[IngestJson]):
    count = ingest_json_service([passage.model_dump() for passage in passages])
    return {"ingested": count}

# Endpoint for similarity search
@router.post("/search-passages")
async def passages_similarity_search(request: SearchRequest):
    return search(request.query, request.k)

# Endpoint for raw text ingestion
@router.post("/ingest-text")
async def ingest_raw_text(request: IngestTextRequest):
    count = ingest_text(request.text)
    return {"ingested_chunks": count}

# LangGraph-powered endpoint with LLM response
@router.post("/search-text")
async def search_text(request: SearchRequest):
    """
    LangGraph-powered RAG endpoint that retrieves from freewriting collection
    and generates an LLM response based on the results.
    """
    result = await search_text_graph.ainvoke(
        {"query": request.query, "k": request.k},
        config={"configurable": {"thread_id": "freewriting_search"}}
    )

    return {
        "answer": result.get("answer"),
        "sources": result.get("docs"),
        "query": request.query
    }

# Endpoint tha uses NER to extract entities from the "freewriting" collection
# @router.post("/ner-search-text")
# async def ner_search_text(request:SearchRequest):
#
#     result = search(request.query, request.k, collection="freewriting")
#
#     combined_text = "\n\n".join(passage["text"] for passage in result)
#
#     entities = extract_entities(combined_text)
#
#     # create a new prompt for the LLM and tell it to help with classification
#     # Another example of RAG - we're retrieving info that will augment the response
#
#     prompt = (
#         f"Based on the following extracted entities from the freewriting sample,"
#         f"{entities}\n"
#         f"Answer the User's NER-based query with ONLY the data you see here."
#         f"User query: {request.query}"
#     )
#
#     return chain.invoke({"input":prompt})