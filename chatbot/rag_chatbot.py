import os
import re
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHAT_MODEL   = "gpt-4o-mini"
TOP_K        = 20         
MEMORY_WINDOW = 6


SYSTEM_TEMPLATE = """You are a helpful faculty information assistant for Jaypee University \
of Information Technology (JUIT), Waknaghat, Himachal Pradesh.

You help students, staff, and visitors find accurate information about JUIT faculty members \
across all departments: Computer Science & IT, Electronics & Communication, Humanities & \
Social Sciences, Biotechnology & Informatics, and Civil Engineering.

Use ONLY the retrieved context below to answer the question. Do NOT invent details.

Rules:
- For listing queries, list EVERY faculty member present in the context — do not truncate.
- For count queries, count exactly how many faculty appear in the context and state that number.
- Format multiple faculty as a numbered or bulleted list with name and designation.
- For contact info add: "Please verify on the official JUIT website: https://www.juit.ac.in"
- If info is not in the context, say so and direct to https://www.juit.ac.in

Retrieved Context:
──────────────────
{context}
──────────────────
"""

HUMAN_TEMPLATE = "{question}"

QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the following conversation history and a follow-up question, "
        "rephrase the follow-up question to be a standalone, self-contained question. "
        "Return ONLY the rephrased question.",
    ),
    ("human", "Chat history:\n{chat_history}\n\nFollow-up: {question}"),
])


DEPT_KEYWORDS: dict[str, list[str]] = {
    "Computer Science & IT": [
        "cse", "cs", "computer science", "information technology",
        "it department", "software", "programming",
    ],
    "Electronics & Communication": [
        "ece", "electronics", "communication", "vlsi", "embedded",
        "signal processing", "electrical",
    ],
    "Humanities & Social Sciences": [
        "humanities", "social science", "english", "economics",
        "management", "sociology", "philosophy",
    ],
    "Biotechnology & Informatics": [
        "biotech", "biotechnology", "bioinformatics", "biology",
        "biochemistry", "genetics",
    ],
    "Civil Engineering": [
        "civil", "construction", "structural", "geotechnical",
        "transportation", "environmental engineering",
    ],
}

AGGREGATE_PATTERNS = [
    r"\ball\b",
    r"\blist\b",
    r"\bhow many\b",
    r"\bcount\b",
    r"\btotal\b",
    r"\beveryone\b",
    r"\bevery\b",
    r"\bfull list\b",
    r"\bcomplete list\b",
    r"\bshow all\b",
    r"\bname all\b",
]


def detect_department(query: str) -> Optional[str]:

    q = query.lower()

    for dept, keywords in DEPT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return dept
    return None


def is_aggregate_query(query: str) -> bool:
    """Return True if the query is asking for a full list or count."""

    q = query.lower()

    return any(re.search(pat, q) for pat in AGGREGATE_PATTERNS)



class JUITChatbot:

    def __init__(
        self,
        vectorstore: Chroma,
        openai_api_key: Optional[str] = None,
        model: str = CHAT_MODEL,
        top_k: int = TOP_K,
    ):
        api_key = openai_api_key or os.environ["OPENAI_API_KEY"]
        self.vectorstore = vectorstore
        self.top_k = top_k

        self.llm = ChatOpenAI(
            model=model,
            temperature=0.2,
            openai_api_key=api_key,
        )

        self.memory = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW,
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
        )


    def _fetch_aggregate(self, department: Optional[str]) -> list[Document]:
        """
        Bypass semantic search: fetch ALL faculty docs for a dept (or all depts)
        directly from ChromaDB using metadata filter.
        """
        from vectorstore.vector_store import get_all_faculty_by_department, get_all_faculty

        if department:
            return get_all_faculty_by_department(self.vectorstore, department)
        else:
            return get_all_faculty(self.vectorstore)

    def _fetch_semantic(self, query: str, department: Optional[str]) -> list[Document]:
        """Standard top-k semantic search, optionally filtered by department."""

        search_kwargs: dict = {"k": self.top_k}

        if department:
            search_kwargs["filter"] = {"department": department}

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

        return retriever.invoke(query)


    def _build_chain(self, retriever) -> ConversationalRetrievalChain:

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            condense_question_prompt=CONDENSE_PROMPT,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
            verbose=False,
        )


    def _answer_aggregate(self, query: str, docs: list[Document]) -> str:
        """
        For aggregate queries, build context from ALL docs and call LLM directly.
        We don't use ConversationalRetrievalChain here because we already have
        all the docs — no retrieval step needed.
        """

        context = "\n\n".join(d.page_content for d in docs)

        messages = QA_PROMPT.format_messages(context=context, question=query)
        response = self.llm.invoke(messages)

        return response.content.strip()


    def chat(self, user_message: str) -> dict:

        logger.info(f"Query: {user_message}")

        dept        = detect_department(user_message)
        aggregate   = is_aggregate_query(user_message)

        logger.info(f"  dept={dept}  aggregate={aggregate}")

        if aggregate:
            docs   = self._fetch_aggregate(dept)
            answer = self._answer_aggregate(user_message, docs)

            self.memory.chat_memory.add_user_message(user_message)
            self.memory.chat_memory.add_ai_message(answer)

            source_docs = docs

        else:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.top_k,
                    **({"filter": {"department": dept}} if dept else {}),
                },
            )
            chain  = self._build_chain(retriever)
            result = chain.invoke({"question": user_message})
            answer = result["answer"]
            source_docs = result.get("source_documents", [])

        seen: set[str] = set()
        sources: list[dict] = []

        for doc in source_docs:
            name = doc.metadata.get("name", "")
            if name and name not in seen:
                seen.add(name)
                sources.append({
                    "name":        name,
                    "department":  doc.metadata.get("department", ""),
                    "designation": doc.metadata.get("designation", ""),
                    "email":       doc.metadata.get("email", ""),
                    "profile_url": doc.metadata.get("profile_url", ""),
                })

        logger.info(f"Answer preview: {answer[:120]}…")
        
        return {
            "answer":           answer,
            "source_documents": source_docs,
            "sources":          sources,
            "department_filter":dept,
            "query_type":       "aggregate" if aggregate else "semantic",
        }

    def reset(self):
        self.memory.clear()
        logger.info("Memory cleared.")



def run_cli(chatbot: JUITChatbot):
    print("\n" + "=" * 60)
    print("  JUIT Faculty Chatbot  (LangChain RAG)")
    print("  'reset' — clear history | 'quit' — exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!"); break
        if user_input.lower() == "reset":
            chatbot.reset()
            print("History cleared.\n"); continue

        result = chatbot.chat(user_input)
        print(f"\nAssistant: {result['answer']}")
        print(f"[{result['query_type']} query | {len(result['sources'])} faculty retrieved]")

        if result["sources"]:
            print("\n📚 Sources:")
            for s in result["sources"][:5]:
                print(f"  • {s['name']} | {s['department']} | {s['designation']}")
        print()