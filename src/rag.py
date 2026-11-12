"""
RAG pipeline for MineGPT.

Ties the retriever and the LLM client together:
  1. Retrieve the most relevant DGMS passages for the question.
  2. Build a grounded prompt from those passages.
  3. Ask the LLM to answer using ONLY that context (to stay factual).
  4. If no LLM is configured, fall back to an extractive answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMClient
from .retriever import Chunk, Retriever, ensure_index

SYSTEM_PROMPT = (
    "You are MineGPT, an assistant specialised in Indian mining rules and "
    "regulations administered by the DGMS (Directorate General of Mines Safety). "
    "Answer the user's question using ONLY the provided context passages. "
    "Be precise and factual; cite the relevant Act/Regulation/Rule and section or "
    "regulation number when the context gives it. "
    "If the context does not contain the answer, say clearly that the provided "
    "DGMS material does not cover it rather than inventing details. "
    "Remind the user to verify against the official Gazette text for legal use when "
    "quoting specific provisions."
)


@dataclass
class Answer:
    text: str
    sources: list[Chunk]
    mode: str  # "llm" or "extractive"


def _format_context(results: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for i, (chunk, score) in enumerate(results, 1):
        blocks.append(
            f"[Passage {i}] (source: {chunk.source} — {chunk.heading})\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def _extractive_answer(question: str, results: list[tuple[Chunk, float]]) -> str:
    """Fallback answer when no LLM is configured: present the top passages."""
    if not results:
        return (
            "I could not find anything relevant to that in the DGMS knowledge base. "
            "Try rephrasing, or ask about the Mines Act 1952, Coal Mines Regulations "
            "2017, Metalliferous Mines Regulations 1961, or mine safety topics."
        )
    lines = ["Based on the DGMS knowledge base, the most relevant provisions are:\n"]
    for chunk, score in results:
        snippet = chunk.text.strip()
        lines.append(f"- ({chunk.source} — {chunk.heading})\n  {snippet}\n")
    lines.append(
        "\n[No LLM configured: showing retrieved passages. Add an OPENROUTER_API_KEY "
        "in .env for conversational answers.]"
    )
    return "\n".join(lines)


class MineGPT:
    """The high-level question-answering object used by the CLI."""

    def __init__(self, rebuild_index: bool = False, top_k: int = 4) -> None:
        self.retriever: Retriever = ensure_index(rebuild=rebuild_index)
        self.llm = LLMClient()
        self.top_k = top_k

    def ask(self, question: str) -> Answer:
        results = self.retriever.search(question, top_k=self.top_k)
        sources = [c for c, _ in results]

        if not self.llm.available:
            return Answer(_extractive_answer(question, results), sources, "extractive")

        if not results:
            return Answer(
                "The provided DGMS material does not appear to cover that question. "
                "Please ask about Indian mining law/safety (DGMS), or rephrase.",
                sources,
                "llm",
            )

        context = _format_context(results)
        user_prompt = (
            f"Context passages from the DGMS knowledge base:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above."
        )
        try:
            text = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        except Exception as exc:  # network/key/model errors -> graceful fallback
            fallback = _extractive_answer(question, results)
            return Answer(
                f"[LLM call failed: {exc}. Falling back to retrieved passages.]\n\n{fallback}",
                sources,
                "extractive",
            )
        return Answer(text, sources, "llm")
