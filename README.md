# MineGPT 🪨⛏️

A simple **RAG (Retrieval-Augmented Generation) chatbot** that answers questions
about **Indian mining rules and regulations** administered by the **DGMS**
(Directorate General of Mines Safety).

It retrieves the most relevant passages from a curated DGMS knowledge base
(Mines Act 1952, Coal Mines Regulations 2017, Metalliferous Mines Regulations
1961, Mines Rules 1955, Rescue & Training Rules, and safety topics) and uses an
LLM (via OpenRouter) to answer **grounded only in that material**, so answers
stay factual.

## How it works

```
your question
     │
     ▼
TF-IDF retriever  ──►  top-k DGMS passages
     │                        │
     └──────────┬─────────────┘
                ▼
     grounded prompt  ──►  LLM (OpenRouter)  ──►  cited answer
```

- **Retrieval:** `scikit-learn` TF-IDF + cosine similarity over chunked documents
  (lightweight, offline, no large downloads).
- **Generation:** OpenRouter (OpenAI-compatible). The model is told to answer
  *only* from the retrieved context and to cite the Act/Regulation/section.
- **No key? Still works:** without an API key it returns the retrieved passages
  (extractive mode).

## Setup

Clone the repo and create a virtual environment (the `.venv/` is not committed):

```bash
git clone https://github.com/Iitian-shivansh/MineGPT.git
cd MineGPT

py -3.12 -m venv .venv            # Windows (Python 3.12 recommended)
.venv\Scripts\activate           # Windows (PowerShell/CMD)
# python3 -m venv .venv && source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Add your OpenRouter key (for conversational answers)

1. Get a free key at https://openrouter.ai/keys
2. Copy `.env.example` to `.env` and paste your key:

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
```

## Usage

```bash
# interactive chat
.venv\Scripts\python.exe chat.py

# single question
.venv\Scripts\python.exe chat.py -q "What are the maximum working hours below ground?"

# rebuild the index after editing data/
.venv\Scripts\python.exe chat.py --rebuild
```

Example questions:
- *What does DGMS do?*
- *What are the working-hour limits above and below ground?*
- *Within what time must a fatal accident be inquired into?*
- *Can women work below ground in a mine?*
- *What is a Safety Management Plan under CMR 2017?*
- *What competency certificates does DGMS grant?*

## Project structure

```
MineGPT/
├── chat.py                # CLI chatbot entry point
├── requirements.txt
├── .env.example           # copy to .env and add your key
├── data/                  # DGMS knowledge base (editable .md files)
│   ├── 01_dgms_overview.md
│   ├── 02_mines_act_1952.md
│   ├── 03_mines_rules_1955.md
│   ├── 04_coal_mines_regulations_2017.md
│   ├── 05_metalliferous_mines_regulations_1961.md
│   ├── 06_rescue_and_training_rules.md
│   ├── 07_personnel_and_competency.md
│   ├── 08_safety_key_topics.md
│   └── 09_faq.md
└── src/
    ├── retriever.py       # TF-IDF index + chunking + search
    ├── llm.py             # OpenRouter client
    └── rag.py             # retrieve + generate pipeline
```

## Extending the knowledge base

Drop any additional `.md` or `.txt` files into `data/` (for example a specific
DGMS circular or regulation you want covered), then run with `--rebuild`. The
retriever automatically picks up new files.

## Disclaimer

MineGPT is an **educational tool**. Its knowledge base is a summary of the law.
For any legal, statutory or operational decision, always rely on the official
**Gazette of India** text of the Act/rules/regulations and current **DGMS**
circulars at [dgms.gov.in](https://www.dgms.gov.in).
