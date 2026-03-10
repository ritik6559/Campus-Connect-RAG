import os
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
MEMORY_WINDOW = 6  

SYSTEM_TEMPLATE = """You are a helpful faculty information assistant for Jaypee University \
of Information Technology (JUIT), Waknaghat, Himachal Pradesh.

You help students, staff, and visitors find accurate information about JUIT faculty members \
across all departments: Computer Science & IT, Electronics & Communication, Humanities & \
Social Sciences, Biotechnology & Informatics, and Civil Engineering.

Use ONLY the retrieved context below to answer the question. If the answer is not found \
in the context, say so clearly and suggest visiting https://www.juit.ac.in

Rules:
- Be concise, accurate, and professional.
- Format multiple faculty as a clean list.
- For contact info always add: "Please verify on the official JUIT website."
- Never invent or guess faculty details.

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
        "rephrase the follow-up question to be a standalone question that contains "
        "all necessary context. Return ONLY the rephrased question.",
    ),
    ("human", "Chat history:\n{chat_history}\n\nFollow-up question: {question}"),
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


def detect_department(query: str) -> Optional[str]:
    """Return the most likely department name from query keywords, or None."""
    q = query.lower()
    for dept, keywords in DEPT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return dept
    return None


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

        self._retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        self._chain = self._build_chain(self._retriever)

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

    def _get_chain_for_query(self, query: str) -> ConversationalRetrievalChain:
        dept = detect_department(query)
        if dept:
            logger.info(f"Department filter applied: {dept}")
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k, "filter": {"department": dept}},
            )
        else:
            retriever = self._retriever

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            condense_question_prompt=CONDENSE_PROMPT,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
            verbose=False,
        )

    def chat(self, user_message: str) -> dict:
       
        logger.info(f"Query: {user_message}")
        chain = self._get_chain_for_query(user_message)
        result = chain.invoke({"question": user_message})

        answer: str = result["answer"]
        source_docs = result.get("source_documents", [])

        seen: set[str] = set()
        sources: list[dict] = []
        for doc in source_docs:
            name = doc.metadata.get("name", "")
            if name and name not in seen:
                seen.add(name)
                sources.append({
                    "name": name,
                    "department": doc.metadata.get("department", ""),
                    "designation": doc.metadata.get("designation", ""),
                    "email": doc.metadata.get("email", ""),
                    "profile_url": doc.metadata.get("profile_url", ""),
                })

        logger.info(f"Answer: {answer[:120]}...")
        return {
            "answer": answer,
            "source_documents": source_docs,
            "sources": sources,
            "department_filter": detect_department(user_message),
        }

    def reset(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Memory cleared.")



def run_cli(chatbot: JUITChatbot):
    print("\n" + "=" * 60)
    print("  JUIT Faculty Information Chatbot  (LangChain RAG)")
    print("  Commands: 'reset' — clear history | 'quit' — exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            chatbot.reset()
            print("Conversation memory cleared.\n")
            continue

        result = chatbot.chat(user_input)
        print(f"\nAssistant: {result['answer']}")

        if result["sources"]:
            print("\n📚 Sources:")
            for s in result["sources"][:4]:
                line = f"  • {s['name']} | {s['department']}"
                if s["designation"]:
                    line += f" | {s['designation']}"
                print(line)
        print()