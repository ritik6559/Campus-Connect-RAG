import os
import sys
import logging
import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_chatbot_instance = None   
_vectorstore      = None   


def _get_chatbot():
    return _chatbot_instance



def chat_fn(user_message: str, history: list):
    """
    Gradio 6.x: history is list[dict] — {"role": "user"/"assistant", "content": str}
    """

    history = history or []

    if not user_message.strip():
        return history, "", _sources_placeholder()

    bot = _get_chatbot()
    if bot is None:

        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content":
                         "Please ingest faculty data first in the **Setup** tab."})
        return history, "", ""

    result = bot.chat(user_message)

    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": result["answer"]})

    return history, "", _format_sources(result)


def reset_fn():
    bot = _get_chatbot()

    if bot:
        bot.reset()

    return [], "", "Conversation memory cleared."


def scrape_and_ingest(api_key: str, progress=gr.Progress()):
    if not api_key.strip():
        return "Enter your OpenAI API key first."

    os.environ["OPENAI_API_KEY"] = api_key.strip()

    try:
        from scraper.faculty_scraper import FacultyScraper
        from vectorstore.vector_store import build_vectorstore, get_collection_stats
        from chatbot.rag_chatbot import JUITChatbot

        progress(0.05, desc="Starting scraper…")
        docs = FacultyScraper(delay=1.5).scrape_all()
        progress(0.50, desc=f"Scraped {len(docs)} faculty. Embedding…")

        global _vectorstore, _chatbot_instance
        _vectorstore      = build_vectorstore(docs, openai_api_key=api_key.strip(), reset=True)
        _chatbot_instance = JUITChatbot(vectorstore=_vectorstore, openai_api_key=api_key.strip())

        progress(1.0, desc="Done!")
        stats = get_collection_stats()
        lines = [
            f"**{sum(stats['by_department'].values())} faculty** indexed\n",
            "**By Department:**",
        ]
        for dept, cnt in stats["by_department"].items():
            lines.append(f"- {dept}: **{cnt}** faculty")
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
        from vectorstore.vector_store import build_from_json, get_collection_stats
        from chatbot.rag_chatbot import JUITChatbot

        progress(0.2, desc="Reading JSON…")
        global _vectorstore, _chatbot_instance
        _vectorstore      = build_from_json(json_file.name, openai_api_key=api_key.strip(), reset=True)
        _chatbot_instance = JUITChatbot(vectorstore=_vectorstore, openai_api_key=api_key.strip())

        progress(1.0, desc="Done!")
        stats = get_collection_stats()
        total = sum(stats["by_department"].values())
        return f"Loaded **{total} faculty** from JSON."

    except Exception as exc:
        logger.exception("JSON load failed")
        return f"Error: {exc}"



def _sources_placeholder() -> str:
    return "*Sources will appear here after your first query.*"


def _format_sources(result: dict) -> str:
    sources = result.get("sources", [])
    q_type  = result.get("query_type", "semantic")

    if not sources:
        return "*No sources retrieved.*"

    lines = [f"**{len(sources)} Faculty Retrieved** *(query type: {q_type})*\n"]
    for s in sources[:10]:
        line = f"- **{s['name']}**"
        if s.get("department"):  line += f" | {s['department']}"
        if s.get("designation"): line += f" | {s['designation']}"
        if s.get("email"):       line += f"\n  📧 `{s['email']}`"
        lines.append(line)

    if len(sources) > 10:
        lines.append(f"\n*…and {len(sources) - 10} more*")

    dept = result.get("department_filter")
    if dept:
        lines.append(f"\n*🔍 Department filter: {dept}*")

    return "\n".join(lines)


EXAMPLES = [
    "List all faculty in CSE department",
    "How many faculty are there in ECE?",
    "Who are the professors in Civil Engineering?",
    "Which faculty specializes in machine learning?",
    "What is the email of the HOD of ECE?",
    "Show me all Assistant Professors in Humanities",
    "How many total faculty are there across all departments?",
    "Which faculty have a PhD from IIT?",
]

CSS = """
    .source-panel  { background:#f5f6fa; border-radius:10px; padding:14px; font-size:0.87em; line-height:1.6; }
    .query-badge   { font-size:0.75em; color:#6366f1; }
    footer         { display:none !important; }
"""

with gr.Blocks(title="JUIT Faculty Chatbot") as demo:

    gr.Markdown("""
    # JUIT Faculty Information Chatbot
    ### LangChain · ChromaDB · OpenAI GPT-4o-mini
    *Ask anything about faculty — names, designations, emails, research areas, counts, and more.*
    """)

    # ── Chat Tab ───────────────────────────────────────────────────────────────
    with gr.Tab("💬 Chat"):
        with gr.Row(equal_height=False):

            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(
                    label="Faculty Assistant",
                    height=500,
                    show_label=True,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="e.g.  List all CSE faculty  /  How many ECE professors?",
                        label="Your Question",
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=90)
                with gr.Row():
                    reset_btn = gr.Button("Reset Conversation", variant="secondary", size="sm")

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=msg_input,
                    label="Example Questions",
                )

            with gr.Column(scale=1, min_width=280):
                sources_panel = gr.Markdown(
                    value=_sources_placeholder(),
                    label="Retrieved Sources",
                    elem_classes=["source-panel"],
                )

        send_btn.click(
            chat_fn,
            inputs=[msg_input, chatbot_ui],
            outputs=[chatbot_ui, msg_input, sources_panel],
        )
        msg_input.submit(
            chat_fn,
            inputs=[msg_input, chatbot_ui],
            outputs=[chatbot_ui, msg_input, sources_panel],
        )
        reset_btn.click(
            reset_fn,
            outputs=[chatbot_ui, msg_input, sources_panel],
        )

    with gr.Tab("⚙️ Setup & Ingest"):
        gr.Markdown("""
        ### One-time Setup
        1. Enter your **OpenAI API key**
        2. Click **Scrape JUIT Website** (fetches live data) or upload a saved JSON
        3. Once ingestion is complete, switch to the **Chat** tab
        """)

        api_key_input = gr.Textbox(
            label="🔑 OpenAI API Key",
            placeholder="sk-…",
            type="password",
            info="Used only for this session — never stored or sent anywhere else.",
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 🌐 Option A — Scrape Live Website")
                gr.Markdown(
                    "Crawls all 5 JUIT department faculty pages in real time "
                    "and builds the ChromaDB index."
                )
                scrape_btn    = gr.Button("Scrape JUIT Website & Ingest", variant="primary")
                scrape_status = gr.Markdown("*Ready.*")
                scrape_btn.click(scrape_and_ingest, inputs=[api_key_input], outputs=scrape_status)

            with gr.Column():
                gr.Markdown("#### 📂 Option B — Upload Pre-scraped JSON")
                gr.Markdown(
                    "Upload `faculty_data.json` from a previous scrape. "
                    "Much faster — no HTTP requests needed."
                )
                json_upload = gr.File(label="faculty_data.json", file_types=[".json"])
                load_btn    = gr.Button("Load JSON & Ingest", variant="secondary")
                load_status = gr.Markdown("*Ready.*")
                load_btn.click(load_json, inputs=[json_upload, api_key_input], outputs=load_status)

    with gr.Tab("ℹ️ How It Works"):
        gr.Markdown("""
        ## RAG Architecture

        ```
        JUIT Website (5 dept pages)
              │  BeautifulSoup scraper
              ▼
        LangChain Documents  (1 doc = 1 faculty member)
              │  OpenAIEmbeddings (text-embedding-3-small)
              ▼
        ChromaDB (cosine similarity index)
              │
              ▼
        User Query ──► Query Router
                            │
                ┌───────────┴────────────────┐
           AGGREGATE?                    SEMANTIC?
        ("list all", "how many"…)    (specific info)
                │                           │
         Metadata filter              Similarity search
         fetch ALL docs               top-20 results
                │                           │
                └───────────┬───────────────┘
                            │
                      GPT-4o-mini
                      + Full Context
                            │
                          Answer
        ```

        ## Why Two Paths?

        Semantic search returns the **most similar** documents (top-k).
        For "list all CSE faculty", the most similar 5 docs are not the same as **all** CSE faculty.
        The aggregate path bypasses similarity and fetches **every doc** matching the department metadata filter directly from ChromaDB.

        ## LangChain Stack

        | Component | Class |
        |---|---|
        | LLM | `ChatOpenAI` (gpt-4o-mini) |
        | Embeddings | `OpenAIEmbeddings` (text-embedding-3-small) |
        | Vector Store | `langchain_chroma.Chroma` |
        | RAG Chain | `ConversationalRetrievalChain` |
        | Memory | `ConversationBufferWindowMemory` |
        | Prompts | `ChatPromptTemplate` |
        | Documents | `langchain_core.documents.Document` |
        """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=CSS,
    )