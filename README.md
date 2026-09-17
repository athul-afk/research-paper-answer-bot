# Research Paper Answer Bot — GenAI Pinnacle Plus Capstone

A RAG (Retrieval-Augmented Generation) chatbot that answers questions over a curated set
of seminal Generative AI research papers, citing the exact paper title and page number
for every claim.

## Project Structure

```
pinnacle_project/
├── notebook/
│   ├── Research_Paper_Answer_Bot.ipynb   <- MAIN DELIVERABLE: full pipeline, run top to bottom
│   ├── requirements.txt
│   └── data/papers/                       <- downloaded PDFs land here (created on first run)
├── app/                                    <- Stretch Goal 2: Streamlit UI
│   ├── app.py
│   ├── rag_core.py
│   └── requirements.txt
├── slides/
│   └── Research_Paper_Answer_Bot_Presentation.pptx
├── requirements.txt                        <- combined deps for everything
└── README.md
```

## How to run

### 1. Notebook (main deliverable)
```bash
cd notebook
pip install -r requirements.txt
jupyter notebook Research_Paper_Answer_Bot.ipynb
```
Run all cells top to bottom ("Restart & Run All"). Works with or without an
`OPENAI_API_KEY` — set it as an environment variable if you have one, otherwise the
notebook automatically falls back to open-source models everywhere a paid API would
normally be used.

### 2. Streamlit app (Stretch Goal 2)
Run the notebook at least once first (it builds the vector index that the app reuses),
then:
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## What's implemented

- **Compulsory goals:** document loading & indexing, 2+ embedding models compared,
  4 retrieval strategies compared (dense, MMR, hybrid BM25, cross-encoder reranker),
  full RAG chain with top-3 cited sources (paper title + page number), tested on 10
  queries.
- **Stretch goals (2 of the required 1):**
  - Option 1 — Conversational memory with query condensation
  - Option 2 — Streamlit UI

## Before your live review

Read through every markdown cell in the notebook — they explain *why* each choice was
made (chunking strategy, embedding model, retrieval strategy, prompt design). You'll be
asked to defend these choices live, so make sure you can explain each one in your own
words, not just recite what's written.
