# Fake News Detection System

An end-to-end fact-checking and veracity verification system built around the official **LIAR benchmark dataset** (Wang, ACL 2017). The system combines machine learning baseline classification, robust input ingestion, FAISS dense evidence retrieval, and automated verification.

---

## Architecture & Phases

- **Phase 1: Dataset Pipeline** ([`src/data/`](file:///c:/Users/pk902/fake_news_detection/src/data)): Official LIAR benchmark ingestion, 6-way and binary label mapping, strict data validation, and zero-leakage split preservation (Train: 10,269, Valid: 1,284, Test: 1,283).
- **Phase 2: Baseline Classification** ([`src/classification/`](file:///c:/Users/pk902/fake_news_detection/src/classification)): Ultra-fast CPU-only TF-IDF (10,000 unigrams + bigrams) with balanced Logistic Regression, Platt probability calibration, and persistence via Joblib.
- **Phase 3: Article & URL Ingestion** ([`src/ingestion/`](file:///c:/Users/pk902/fake_news_detection/src/ingestion)): Multi-format input handling (claims, headlines, raw articles, web URLs), readability extraction with `trafilatura` and `beautifulsoup4`, SSRF protection, length bounds, and error recovery.
- **Phase 4: Evidence Retrieval** ([`src/retrieval/`](file:///c:/Users/pk902/fake_news_detection/src/retrieval)): Curated institutional reference corpus, SentenceTransformers embeddings (`all-MiniLM-L6-v2`), FAISS vector index, and supporting vs. contradicting passage categorization.

---

## Quickstart

### 1. Run the Full Test Suite (43 Unit Tests)
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### 2. Ingest and Prepare the LIAR Dataset
```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py
```

### 3. Train the Baseline Classification Model
```powershell
.\.venv\Scripts\python.exe scripts/train_model.py --model-type baseline
```

### 4. Build the FAISS Evidence Index
```powershell
.\.venv\Scripts\python.exe scripts/build_evidence_index.py
```

### 5. Analyze Real User Inputs with Evidence Retrieval

#### Analyze a Claim with Evidence Retrieval
```powershell
.\.venv\Scripts\python.exe scripts/analyze.py "The Apollo moon landing was staged in a Hollywood television studio."
```

#### Analyze a Scientific Fact
```powershell
.\.venv\Scripts\python.exe scripts/analyze.py "NASA rovers and radar detected water ice on Mars."
```

#### Analyze a Live Web URL (with SSRF Defense & Boilerplate Removal)
```powershell
.\.venv\Scripts\python.exe scripts/analyze.py "https://www.reuters.com/world/article-example"
```

---

## Phase 4: Evidence Retrieval Architecture

| Component | File | Description |
| :--- | :--- | :--- |
| **Reference Corpus** | [`data/evidence/reference_documents.json`](file:///c:/Users/pk902/fake_news_detection/data/evidence/reference_documents.json) | High-trust institutional knowledge from WHO, CDC, NASA, IPCC, NOAA, BLS, and FCC |
| **Passage Chunking** | [`src/preprocessing/chunking.py`](file:///c:/Users/pk902/fake_news_detection/src/preprocessing/chunking.py) | Sentence-boundary aligned passage chunking with contextual overlap |
| **Embeddings** | [`src/retrieval/embeddings.py`](file:///c:/Users/pk902/fake_news_detection/src/retrieval/embeddings.py) | Free local CPU `all-MiniLM-L6-v2` dense embeddings (384-dim, normalized) |
| **FAISS Vector Store** | [`src/retrieval/vector_store.py`](file:///c:/Users/pk902/fake_news_detection/src/retrieval/vector_store.py) | Inner Product / Cosine similarity search with disk persistence |
| **Stance Analysis** | [`src/retrieval/search.py`](file:///c:/Users/pk902/fake_news_detection/src/retrieval/search.py) | Categorizes evidence into `SUPPORTS`, `CONTRADICTS`, or `NEUTRAL` |
| **Source Credibility** | [`src/retrieval/source_validation.py`](file:///c:/Users/pk902/fake_news_detection/src/retrieval/source_validation.py) | Ranks sources by authoritative credibility tiers (Tier 1 to Tier 3) |
