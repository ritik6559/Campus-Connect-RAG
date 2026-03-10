from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os
import argparse, sys

PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "faculty_name"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

def _split_documents(docs: List[Document]) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(docs)
    print(f"Splitted into {len(chunks)} chunks.")
    print(f"First chunk: {chunks[0]}.")

    return chunks

def json_parsing(filepath: str) -> List[Document]:

    with open(filepath, 'r') as f:
        data = json.load(f)
    
    documents = []

    for emp in data:
        content = emp['page_content']
        metadata = emp['metadata']

        doc = Document(
            page_content = content,
            metadata = metadata
        )

        documents.append(doc)

    return documents
        
def load_vectorstore(
    chroma_path: str = PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    1
    embeddings = HuggingFaceEmbeddings()
    return Chroma(
        collection_name = collection_name,
        embedding_function = embeddings,
        persist_directory = chroma_path,
    )

def build_vector_store(documents: List[Document]):
    
    chunks = _split_documents(documents)

    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = HuggingFaceEmbeddings(),
        persist_directory = PERSIST_DIRECTORY,
        collection_name = COLLECTION_NAME
    )

    print(f"Vector store created with {vector_store._collection.count()} vectors")
    print(f"Persisted to: {PERSIST_DIRECTORY}")

    return vector_store

def get_retriever(vectorstore: Chroma):

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

def get_collection_stats(
    chroma_path: str = PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME
) -> dict:

    embeddings = HuggingFaceEmbeddings()

    vectorstore = Chroma(
        persist_directory=chroma_path,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    total = vectorstore._collection.count()

    by_dept = {}

    if total:
        result = vectorstore._collection.get(include=["metadatas"])

        for m in result["metadatas"]:
            dept = m.get("department", "Unknown")
            by_dept[dept] = by_dept.get(dept, 0) + 1

    return {"total_chunks": total, "by_department": by_dept}

if __name__ == "__main__":
    sys.path.append("..")

    parser = argparse.ArgumentParser(description="Build JUIT ChromaDB vector store")
    parser.add_argument("--from-json", metavar="PATH", help="Load from saved JSON")
    parser.add_argument("--from-web", action="store_true", help="Scrape live")
    args = parser.parse_args()

    if args.from_json:
        vs = json_parsing(args.from_json)
    elif args.from_web:
        from scraper import FacultyScraper
        
        docs = FacultyScraper().scrape_all()
        vs = build_vector_store(docs)
    else:

        parser.print_help()
        sys.exit(1)

    stats = get_collection_stats()
    
    print(f"\nDone! Total chunks: {stats['total_chunks']}")
    for dept, cnt in stats["by_department"].items():
        print(f"  {dept}: {cnt} chunks")