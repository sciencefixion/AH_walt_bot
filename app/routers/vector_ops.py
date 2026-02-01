from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.routers import passages
from app.services.vectordb_service import ingest_json_service, search, ingest_text, extract_entities

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
    k:int = 3

# TODO: Not sure if the following line is necessary
# chain = get_general_chain()

# Endpoint for data ingestion
@router.post("/ingest-json")
async def ingest_json_endpoint(passages:list[IngestJson]):

    count = ingest_json_service([passage.model_dump() for passage in passages])
    return {"ingested":count}

# Endpoint for similarity search
@router.post("/search-passages")
async def passages_similarity_search(request:SearchRequest):
    return search(request.query, request.k)

# Endpoint for raw text ingestion
@router.post("/ingest-text")
async def ingest_raw_text(request:IngestTextRequest):
    count = ingest_text(request.text)
    return {"ingested_chunks":count}

# Endpoint with LLM-powered response that uses the freewriting collection
# @router.post("/search-text")
# async def search_text(request:SearchRequest):
#
#     # extract results from vector DB
#     result = search(request.query, request.k, collection="freewriting")
#
#     # Quick prompt that tells the LLM the returned results
#     # and asks for it to answer the user based on those results
#     prompt = (
#         f"Based on the following extracted info from the freewriting sample,"
#         f"Answer the user's query to the best of your ability."
#         f"If there's no relevant information stored, you can say that."
#         f"Extracted info: {result}"
#         f"User query: {request.query}"
#     )


# TODO: HOW should langgraph come in here? and also below \/

#     return chain.invoke({"input":prompt})


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