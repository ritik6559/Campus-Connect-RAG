"""
main.py — JUIT Faculty RAG Chatbot (LangChain edition)

Usage examples:
    python main.py --ui                          # Gradio web UI
    python main.py --scrape                      # scrape → embed → CLI chat
    python main.py --scrape --save-json          # scrape → save JSON → embed → CLI chat
    python main.py --from-json faculty_data.json # load JSON → embed → CLI chat
    python main.py --no-ingest                   # use existing ChromaDB → CLI chat
"""

import os
import sys
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ensure_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        key = input("OpenAI API key: ").strip()
        if not key:
            sys.exit("ERROR: OPENAI_API_KEY is required.")
        os.environ["OPENAI_API_KEY"] = key


def main():
    parser = argparse.ArgumentParser(description="JUIT Faculty RAG Chatbot — LangChain")
    parser.add_argument("--scrape", action="store_true", help="Scrape JUIT website")
    parser.add_argument("--from-json", metavar="FILE", help="Load from JSON")
    parser.add_argument("--save-json", action="store_true", help="Save scraped data to JSON")
    parser.add_argument("--no-ingest", action="store_true", help="Skip ingest, use existing ChromaDB")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI")
    args = parser.parse_args()

    if args.ui:
        import subprocess
        subprocess.run([sys.executable, "app.py"])
        return

    ensure_api_key()

    from vectorstore.vector_store import (
        build_vectorstore, build_from_json, load_vectorstore, get_collection_stats
    )
    from chatbot.rag_chatbot import JUITChatbot, run_cli

    if args.no_ingest:
        vectorstore = load_vectorstore()
        stats = get_collection_stats()
        print(f"Loaded existing ChromaDB: {stats['total_chunks']} chunks")

    elif args.from_json:
        print(f"Loading from {args.from_json}…")
        vectorstore = build_from_json(args.from_json, reset=True)
        stats = get_collection_stats()
        print(f"{stats['total_chunks']} chunks indexed")

    elif args.scrape:
        from scraper.faculty_scraper import FacultyScraper
        print("Scraping JUIT faculty website…")
        docs = FacultyScraper(delay=1.5).scrape_all()
        print(f"Scraped {len(docs)} faculty profiles.")

        if args.save_json:
            path = "faculty_data.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
                    f, indent=2, ensure_ascii=False,
                )
            print(f"Saved to {path}")

        vectorstore = build_vectorstore(docs, reset=True)
        stats = get_collection_stats()
        print(f"{stats['total_chunks']} chunks indexed")
        for dept, cnt in stats["by_department"].items():
            print(f"   {dept}: {cnt}")

    else:
        parser.print_help()
        sys.exit(1)

    chatbot = JUITChatbot(vectorstore=vectorstore)
    run_cli(chatbot)


if __name__ == "__main__":
    main()