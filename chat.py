"""
MineGPT — command-line chatbot for Indian mining rules & regulations (DGMS).

Usage:
    python chat.py                 # interactive chat
    python chat.py --rebuild       # rebuild the index, then chat
    python chat.py -q "your question here"   # single question, then exit
    python chat.py --no-sources    # hide the sources panel

Requires an OPENROUTER_API_KEY in a local .env file for conversational answers.
Without it, MineGPT still runs and returns the most relevant DGMS passages.
"""

from __future__ import annotations

import argparse
import sys

# Force UTF-8 output so Unicode (bullets, em-dashes) renders on Windows consoles.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.rag import MineGPT

console = Console()

BANNER = """[bold cyan]MineGPT[/bold cyan] — Indian Mining Rules & Regulations assistant (DGMS)
Ask questions about the Mines Act 1952, Coal Mines Regulations 2017,
Metalliferous Mines Regulations 1961, mine safety, personnel & more.
Type [bold]exit[/bold] or [bold]quit[/bold] to leave.
"""


def render_answer(console: Console, answer, show_sources: bool) -> None:
    console.print(Panel(Markdown(answer.text), title="MineGPT", border_style="cyan"))
    if show_sources and answer.sources:
        seen = []
        for c in answer.sources:
            tag = f"{c.source} — {c.heading}"
            if tag not in seen:
                seen.append(tag)
        src_text = "\n".join(f"• {s}" for s in seen)
        console.print(
            Panel(src_text, title=f"Sources ({answer.mode})", border_style="dim")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MineGPT DGMS RAG chatbot")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the TF-IDF index")
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("--no-sources", action="store_true", help="hide the sources panel")
    parser.add_argument("--top-k", type=int, default=4, help="passages to retrieve")
    args = parser.parse_args()

    with console.status("[cyan]Loading MineGPT knowledge base..."):
        bot = MineGPT(rebuild_index=args.rebuild, top_k=args.top_k)

    mode = "LLM (OpenRouter)" if bot.llm.available else "extractive (no API key)"
    console.print(f"[dim]Index: {len(bot.retriever.chunks)} chunks · Answer mode: {mode}[/dim]\n")

    show_sources = not args.no_sources

    # Single-question mode
    if args.question:
        answer = bot.ask(args.question)
        render_answer(console, answer, show_sources)
        return

    # Interactive mode
    console.print(Panel(BANNER, border_style="cyan"))
    while True:
        try:
            question = console.input("[bold green]You >[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            console.print("[dim]Goodbye.[/dim]")
            break
        with console.status("[cyan]Thinking..."):
            answer = bot.ask(question)
        render_answer(console, answer, show_sources)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
