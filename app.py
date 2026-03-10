import os
import sys
import logging

import gradio as gr

from scraper.faculty_scraper import FacultyScraper
from vectorstore.vector_store import (
    build_vector_store,
    json_parsing,
    get_collection_stats
)
from chatbot.rag_chatbot import JUITChatbot

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_chatbot_instance = None
_vectorstore = None


def _get_chatbot():
    return _chatbot_instance

def chat_fn(user_message: str, history: list):
    history = history or []

    if not user_message.strip():
        return history, "", _sources_placeholder()

    bot = _get_chatbot()
    if bot is None:
        history.append({"role": "user", "content": user_message})
        history.append({
            "role": "assistant",
            "content": "Please ingest faculty data first in the **Setup** tab."
        })
        return history, "", ""

    result = bot.chat(user_message)

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": result["answer"]})

    return history, "", _format_sources(result)


def reset_fn():
    bot = _get_chatbot()
    if bot:
        bot.reset()
    return [], "", "Memory cleared."


def scrape_and_ingest(api_key: str, progress=gr.Progress()):
    if not api_key.strip():
        return "Enter your OpenAI API key first."

    os.environ["OPENAI_API_KEY"] = api_key.strip()

    try:
        progress(0.05, desc="Starting scraper…")
        docs = FacultyScraper(delay=1.5).scrape_all()

        progress(0.55, desc=f"Embedding {len(docs)} documents…")

        global _vectorstore, _chatbot_instance
        _vectorstore = build_vector_store(
            docs,
        )

        _chatbot_instance = JUITChatbot(
            vectorstore=_vectorstore,
            openai_api_key=api_key.strip()
        )

        progress(1.0, desc="Done!")

        stats = get_collection_stats()

        lines = [
            f"**{stats['total_chunks']} chunks** indexed from **{len(docs)} faculty profiles**\n",
            "**By Department:**",
        ]

        for dept, cnt in stats["by_department"].items():
            lines.append(f"- {dept}: {cnt} chunks")

        return "\n".join(lines)

    except Exception as exc:
        logger.exception("Scrape+ingest failed")
        return f"Error: {exc}"


def load_json(json_file, api_key: str, progress=gr.Progress()):
    if not api_key.strip():
        return "Enter your OpenAI API key first."

    if json_file is None:
        return "Upload a faculty_data.json file."

    os.environ["OPENAI_API_KEY"] = api_key.strip()

    try:
        progress(0.2, desc="Reading JSON…")

        global _vectorstore, _chatbot_instance

        _vectorstore = json_parsing(
            json_file.name,
            openai_api_key=api_key.strip(),
            reset=True
        )

        _chatbot_instance = JUITChatbot(
            vectorstore=_vectorstore,
            openai_api_key=api_key.strip()
        )

        progress(1.0, desc="Done!")

        stats = get_collection_stats()

        return f"Loaded **{stats['total_chunks']} chunks** from JSON."

    except Exception as exc:
        logger.exception("JSON load failed")
        return f"Error: {exc}"

def _sources_placeholder():
    return "*Sources will appear here after your first query.*"


def _format_sources(result: dict) -> str:
    sources = result.get("sources", [])

    if not sources:
        return "*No sources retrieved.*"

    lines = ["**Retrieved Faculty Records:**\n"]

    for s in sources[:5]:
        line = f"- **{s['name']}**"

        if s.get("department"):
            line += f" | {s['department']}"

        if s.get("designation"):
            line += f" | {s['designation']}"

        if s.get("email"):
            line += f" | `{s['email']}`"

        lines.append(line)

    dept = result.get("department_filter")

    if dept:
        lines.append(f"\n*Department filter applied: {dept}*")

    return "\n".join(lines)

EXAMPLES = [
    "Who are the professors in the CSE department?",
    "List all faculty in Civil Engineering",
    "Which faculty work on machine learning research?",
    "Who is the HOD of ECE and what is their email?",
    "Show me Assistant Professors in Humanities",
    "Which faculty have specialization in VLSI?",
    "How many faculty are there in each department?",
    "Tell me about faculty with PhD in Biotechnology",
]

CSS = """
.source-panel {
    background: #f5f6fa;
    border-radius: 10px;
    padding: 12px;
    font-size: 0.88em;
}

footer {
    display: none !important;
}
"""



with gr.Blocks(title="JUIT Faculty Chatbot") as demo:

    gr.Markdown("""
# JUIT Faculty Information Chatbot
### Powered by LangChain · ChromaDB · OpenAI GPT-4o-mini
*Ask anything about faculty across all JUIT departments.*
""")

    # ── Chat Tab ───────────────────────────────────────────────────────────────
    with gr.Tab("Chat"):

        with gr.Row(equal_height=False):

            with gr.Column(scale=3):

                chatbot_ui = gr.Chatbot(
                    label="Faculty Assistant",
                    height=480,
                    show_label=True,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="e.g. Who teaches machine learning in CSE?",
                        label="Your Question",
                        scale=5,
                        container=False,
                    )

                    send_btn = gr.Button(
                        "Send ➤",
                        variant="primary",
                        scale=1,
                        min_width=80
                    )

                with gr.Row():
                    reset_btn = gr.Button(
                        "Reset Conversation",
                        variant="secondary",
                        size="sm"
                    )

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=msg_input,
                    label="Example Questions"
                )

            with gr.Column(scale=1, min_width=260):

                sources_panel = gr.Markdown(
                    value=_sources_placeholder(),
                    label="Retrieved Context",
                    elem_classes=["source-panel"],
                )

        send_btn.click(
            chat_fn,
            [msg_input, chatbot_ui],
            [chatbot_ui, msg_input, sources_panel]
        )

        msg_input.submit(
            chat_fn,
            [msg_input, chatbot_ui],
            [chatbot_ui, msg_input, sources_panel]
        )

        reset_btn.click(
            reset_fn,
            outputs=[chatbot_ui, msg_input, sources_panel]
        )

    with gr.Tab("Setup & Ingest"):

        gr.Markdown("""
### One-time Setup

1. Paste your OpenAI API key below  
2. Click **Scrape JUIT & Ingest** (live) or upload a saved JSON  
3. Switch to the Chat tab and start asking questions!
""")

        api_key_input = gr.Textbox(
            label="OpenAI API Key",
            placeholder="sk-…",
            type="password",
            info="Used only for this session, never stored."
        )

        with gr.Row():

            with gr.Column():

                gr.Markdown("#### Option A — Scrape Live Website")

                gr.Markdown(
                    "Crawls all 5 JUIT department pages and indexes faculty into ChromaDB."
                )

                scrape_btn = gr.Button(
                    "Scrape JUIT & Ingest",
                    variant="primary"
                )

                scrape_status = gr.Markdown("*Ready.*")

                scrape_btn.click(
                    scrape_and_ingest,
                    [api_key_input],
                    scrape_status
                )

            with gr.Column():

                gr.Markdown("#### Option B — Upload Pre-scraped JSON")

                gr.Markdown(
                    "Upload `faculty_data.json` from a previous scrape run."
                )

                json_upload = gr.File(
                    label="faculty_data.json",
                    file_types=[".json"]
                )

                load_btn = gr.Button(
                    "Load JSON & Ingest",
                    variant="secondary"
                )

                load_status = gr.Markdown("*Ready.*")

                load_btn.click(
                    load_json,
                    [json_upload, api_key_input],
                    load_status
                )

    with gr.Tab("ℹArchitecture"):

        gr.Markdown("""
## LangChain RAG Pipeline

User Question  
↓  
ConversationalRetrievalChain  
├─ Condense follow-up → standalone question (ChatOpenAI)  
├─ Chroma retriever → top-5 faculty chunks  
└─ GPT-4o-mini → Answer

---

## Stack

| Layer | Library |
|------|--------|
| LLM | langchain_openai.ChatOpenAI |
| Embeddings | OpenAIEmbeddings |
| Vector DB | ChromaDB |
| Chain | ConversationalRetrievalChain |
| Memory | ConversationBufferWindowMemory |
| UI | Gradio 6 |
""")



if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=CSS,
    )