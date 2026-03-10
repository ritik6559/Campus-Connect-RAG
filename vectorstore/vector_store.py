import os
import json
import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "juit_faculty"
EMBED_MODEL     = "text-embedding-3-small"

def _build_embeddings(api_key: Optional[str] = None) -> OpenAIEmbeddings:

    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=api_key or os.environ["OPENAI_API_KEY"],
    )

def build_vectorstore(
    documents: list[Document],
    openai_api_key: Optional[str] = None,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    reset: bool = True,
) -> Chroma:
    """
    Embed and store faculty Documents in ChromaDB.
    """

    embeddings = _build_embeddings(openai_api_key)

    if reset:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_path)
        try:
            client.delete_collection(collection_name)
            logger.info(f"Dropped existing collection '{collection_name}'")
        except Exception:
            pass

    logger.info(f"Embedding {len(documents)} faculty documents…")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=chroma_path,
        collection_metadata={"hnsw:space": "cosine"},
    )
    
    logger.info(f"Vector store ready — {vectorstore._collection.count()} documents.")
    
    return vectorstore


def load_vectorstore(
    openai_api_key: Optional[str] = None,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Load an existing ChromaDB collection (no re-embedding)."""

    embeddings = _build_embeddings(openai_api_key)

    logger.info(f"Loading vector store from '{chroma_path}'…")

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_path,
    )


def build_from_json(
    json_path: str,
    openai_api_key: Optional[str] = None,
    **kwargs,
) -> Chroma:
    """Load Documents from a saved JSON file and build the vector store."""

    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    docs = [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in raw]
    logger.info(f"Loaded {len(docs)} documents from {json_path}")

    return build_vectorstore(docs, openai_api_key=openai_api_key, **kwargs)



def get_all_faculty_by_department(
    vectorstore: Chroma,
    department: str,
) -> list[Document]:
    """
    Fetch EVERY faculty document for a department via metadata filter.
    """
    raw = vectorstore._collection.get(
        where={"department": {"$eq": department}},
        include=["documents", "metadatas"],
    )

    seen: set[str] = set()
    docs: list[Document] = []

    for content, meta in zip(raw["documents"], raw["metadatas"]):
        name = meta.get("name", "")
        if name and name not in seen:
            seen.add(name)
            docs.append(Document(page_content=content, metadata=meta))

    logger.info(f"Direct fetch: {len(docs)} unique faculty in '{department}'")

    return docs


def get_all_faculty(vectorstore: Chroma) -> list[Document]:
    """Fetch ALL faculty documents across every department."""

    raw = vectorstore._collection.get(include=["documents", "metadatas"])
    seen: set[str] = set()
    docs: list[Document] = []

    for content, meta in zip(raw["documents"], raw["metadatas"]):
        name = meta.get("name", "")
        if name and name not in seen:
            seen.add(name)
            docs.append(Document(page_content=content, metadata=meta))

    logger.info(f"Direct fetch: {len(docs)} total unique faculty")

    return docs

def get_collection_stats(
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """Return total document count and unique faculty per department."""

    import chromadb

    client = chromadb.PersistentClient(path=chroma_path)
    
    try:

        col = client.get_collection(collection_name)
    except Exception:
        return {"total_chunks": 0, "by_department": {}}

    total = col.count()
    by_dept: dict[str, int] = {}

    if total:
        result = col.get(limit=total, include=["metadatas"])
        seen: set[str] = set()
        for m in result["metadatas"]:
            name = m.get("name", "")
            dept = m.get("department", "Unknown")
            if name and name not in seen:
                seen.add(name)
                by_dept[dept] = by_dept.get(dept, 0) + 1

    return {"total_chunks": total, "by_department": by_dept}

if __name__ == "__main__":
    import argparse, sys
    
    sys.path.append("..")

    parser = argparse.ArgumentParser(description="Build JUIT ChromaDB vector store")
    parser.add_argument("--from-json", metavar="PATH", help="Load from saved JSON")
    parser.add_argument("--from-web",  action="store_true", help="Scrape live")
    args = parser.parse_args()

    if args.from_json:

        vs = build_from_json(args.from_json)
    elif args.from_web:

        from scraper.faculty_scraper import FacultyScraper
        docs = FacultyScraper().scrape_all()
        vs = build_vectorstore(docs)
    else:
        parser.print_help()
        sys.exit(1)

    stats = get_collection_stats()

    print(f"\nTotal faculty: {sum(stats['by_department'].values())}")

    for dept, cnt in stats["by_department"].items():
        print(f"  {dept}: {cnt}")