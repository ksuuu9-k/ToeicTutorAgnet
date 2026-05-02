# Toeic Tutor Agent

An AI-powered TOEIC tutoring system that goes beyond simple Q&A. Instead of just giving answers, it guides learners through a structured tutoring flow based on their attempt history — from exploratory questions to step-by-step scaffolding to full explanations.

Built as an MVP prototype targeting TOEIC Part 5 (grammar & vocabulary).

---

## Features

- **Adaptive Tutoring State Machine** — PROBE → HINT → SCAFFOLD → REVEAL progression based on attempt count
- **Graph RAG** — Neo4j graph connecting questions, grammar rules, vocabulary, and paraphrases
- **Hybrid Retriever** — combines vector similarity search (OpenAI embeddings) with Cypher-based structured queries
- **Learner History** — SQLite tracks attempts and error patterns per question
- **Conversational UI** — Streamlit chat interface with full conversation context passed to the model

---

## Architecture

```
[Streamlit UI]
      ↓
[Tutor Agent]
  ├── Tutoring State Machine (PROBE → HINT → SCAFFOLD → REVEAL)
  ├── Learner Profile (attempt count, error type — SQLite)
  └── Graph RAG
        ├── Vector Retriever  (question embedding similarity)
        ├── Cypher Retriever  (grammar / difficulty / history filter)
        └── Neo4j
              ├── (:Question)
              ├── (:GrammarRule)
              ├── (:Vocab)
              └── (:Paraphrase)
```

### Tutoring States

| State | Trigger | Behavior |
|-------|---------|----------|
| `PROBE` | 1st wrong attempt | Asks an exploratory question about the sentence |
| `HINT` | 2nd wrong attempt | Provides a keyword or grammar hint |
| `SCAFFOLD` | 3rd wrong attempt | Guides step-by-step reasoning |
| `REVEAL` | 4th+ attempt or explicit request | Reveals answer with full explanation |

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| LLM | Claude Haiku (Anthropic) |
| Graph DB | Neo4j Aura Free |
| Embeddings | OpenAI `text-embedding-3-small` |
| Learner History | SQLite |
| UI | Streamlit |

---

## Project Structure

```
toeic-tuter-agent/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── core/
│   ├── agent.py            # TutorAgent — main orchestrator
│   ├── retriever.py        # VectorRetriever, CypherRetriever, HybridRetriever
│   ├── state_machine.py    # TutoringStateMachine + prompt builder
│   └── learner_db.py       # SQLite learner history
├── data/
│   ├── toeic_100.json
│   └── toeic_100_with_metadata.json
├── scripts/
│   ├── load_graph.py       # Load nodes/edges into Neo4j
│   ├── build_embeddings.py # Generate and store question embeddings
│   └── generate_metadata.py
└── .streamlit/
    └── secrets.toml.example
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your credentials:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
```

> **Neo4j**: Create a free instance at [console.neo4j.io](https://console.neo4j.io) (AuraDB Free).

### 3. Prepare data

The `data/` directory is not included in this repository. Download the source dataset from Kaggle and run the preprocessing scripts:

**Source**: [TOEIC Test Dataset — Kaggle](https://www.kaggle.com/datasets/tientd95/toeic-test)

1. Download and place the raw data in `data/`
2. Run `scripts/generate_metadata.py` to generate `toeic_100_with_metadata.json`

Refer to `scripts/generate_metadata.py` for the expected schema.

### 4. Load data into Neo4j

```bash
python scripts/load_graph.py
```

### 4. Build embeddings

```bash
python scripts/build_embeddings.py
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push this repository to a **private** GitHub repo
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and connect your repo
3. Set `app.py` as the entry point
4. In **App Settings → Secrets**, paste the contents of `.streamlit/secrets.toml.example` with your actual values

---

## Usage

| Input | Action |
|-------|--------|
| `문제 줘` | Get a new question |
| `1` / `2` / `3` / `4` | Submit an answer |
| `힌트 줄래?` | Ask for a hint (free-form) |
| `정답 알려줘` | Skip to full explanation |
