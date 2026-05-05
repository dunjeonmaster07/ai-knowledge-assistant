"""
RAG chain: retriever → prompt → LLM → answer with citations.

The prompt is intentionally generic — this system works with any
documents the user places in data/docs/, not just a specific domain.
"""

import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.config import LLM_MODEL
from src.retriever import get_retriever


PROMPT = ChatPromptTemplate.from_template(
    """You are a knowledgeable assistant that answers questions using ONLY the
context provided below. The context comes from documents uploaded by the user.

Rules:
- Provide detailed, well-structured answers. Use headings, bullet points, or
  numbered steps where appropriate to make the answer easy to read.
- Synthesize information from multiple parts of the context to form a complete answer.
- Cite specific details, sections, or quotes from the context when possible.
- Keep answers concise but thorough — aim for completeness under 500 words.
- If the context does not contain enough information to answer the question,
  say "I don't have enough information in the uploaded documents to answer this."
- Do NOT use your own training knowledge to fill gaps.

Context:
{context}

Question: {question}"""
)


def get_llm(api_key: str | None = None) -> ChatGroq:
    return ChatGroq(model=LLM_MODEL, api_key=api_key) if api_key else ChatGroq(model=LLM_MODEL)


def ask_question(question: str, api_key: str | None = None) -> tuple[str | None, list]:
    """
    Run the full RAG pipeline for a single question.

    Returns:
        (answer_text, retrieved_docs)
    """
    retriever = get_retriever()
    llm = get_llm(api_key=api_key)
    chain = PROMPT | llm

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return None, []

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    response = chain.invoke({"context": context, "question": question})

    return response.content, retrieved_docs


def ask_question_for_eval(question: str) -> dict:
    """
    Same RAG pipeline, but returns the structured dict that RAGAS expects.
    """
    retriever = get_retriever()
    llm = get_llm()
    chain = PROMPT | llm

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "user_input": question,
            "response": "I don't have enough information to answer this.",
            "retrieved_contexts": [],
        }

    contexts = [doc.page_content for doc in retrieved_docs]
    context_str = "\n\n".join(contexts)
    response = chain.invoke({"context": context_str, "question": question})

    return {
        "user_input": question,
        "response": response.content,
        "retrieved_contexts": contexts,
    }
