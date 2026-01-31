import hashlib
from typing import Any

import spacy
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.routers import passages

PERSIST_DIRECTORY = "app/chroma_store"
COLLECTION = "passage_archive"
EMBEDDING = OllamaEmbeddings(model="nomic-embed-text")


vector_store: dict[str, Chroma] = {}


def get_vector_store(collection:str = COLLECTION) -> Chroma:

    if collection not in vector_store:
        vector_store[collection] = Chroma(
            collection_name=collection,
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDING
        )
    return vector_store[collection]


def ingest_json(passages: list[dict[str, Any]], collection:str = COLLECTION) -> int:

    db_instance = get_vector_store(collection)
    docs = [
        Document(page_content=passage["text"], metadata=passage.get("metadata", {}))
    ]
    ids = [passage["id"] for passage in passages]

    db_instance.add_documents(docs, ids=ids)
    return len(passages)

def ingest_text(text:str) -> int:

    text = text.strip()
    if not text:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)

    passages = []

    for index, chunk in enumerate(chunks):

        content_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()[:8]
        chunk_id = f"chunk_{content_hash}"

        passages.append({
            "id": chunk_id,
            "text": chunk,
            "metadata": {
                "chunk_index": index,
                "source":"raw_text_ingestion"
            }
        })

    return ingest_json(passages, collection="freewriting")

def search(query: str, k: int = 10, collection:str = COLLECTION) -> list[dict[str, Any]]:

    db_instance = get_vector_store(collection)

    results = db_instance.similarity_search_with_score(query, k=k)

    return [
        {
            "text": result[0].page_content,
            "metadata": result[0].metadata,
            "score": result[1]
        }
        for result in results
    ]

def extract_entities(text:str):

    ner_model = spacy.load("en_core_web_sm")

    doc = ner_model(text)

    entities = [
        {"text":entity.text, "label":entity.label_}
        for entity in doc.ents
    ]

    return entities