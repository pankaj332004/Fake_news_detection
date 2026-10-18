# Veritas AI — Fake News Detection & Fact Verification System

An end-to-end, multi-stage fact-checking and veracity verification system built around the official **LIAR benchmark dataset** (Wang, ACL 2017). The platform integrates machine learning baseline classification, robust web ingestion, FAISS dense semantic evidence retrieval, open-weight LLM fact verification, uncertainty handling, and an interactive Streamlit web application.

---

## Architecture & Phases

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Ingestion (Phase 3)                        │
│               Claim  •  Headline  •  Full Article  •  URL              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         │                                                     │
┌────────▼─────────────────────────┐        ┌──────────────────▼─────────────────┐
│     Phase 2: ML Baseline         │        │    Phase 4: Semantic Search        │
│   TF-IDF + Logistic Regression   │        │    FAISS Index + MiniLM (384-d)    │
│   Stylistic & Linguistic Cues    │        │  Authoritative Reference Passages  │
└────────────────┬─────────────────┘        └──────────────────┬─────────────────┘
                 │                                             │
                 │                          ┌──────────────────▼─────────────────┐
                 │                          │      Phase 5: LLM Verification     │
                 │                          │    Open-Weight Flan-T5 Reasoning   │
                 │                          │  Contradiction vs Missing Evidence │
                 │                          └──────────────────┬─────────────────┘
                 │                                             │
                 └──────────────────────────┬──────────────────┘
                                            │
                         ┌──────────────────▼──────────────────┐
                         │   Phase 6: Hybrid Decision Policy   │
                         │    Uncertainty Bounds & Consensus   │
                         │   Human-in-the-Loop Review Flagging │
                         └──────────────────┬──────────────────┘
                                            │
                         ┌──────────────────▼──────────────────┐
                         │      Interactive Web Dashboard      │
                         │    app.py (Streamlit Glassmorphic)  │
                         └─────────────────────────────────────┘
```

- **Phase 1: Dataset Pipeline** ([`src/data/`](file:///c:/Users/pk902/fake_news_detection/src/data)): Official LIAR benchmark ingestion, 6-way and binary label mapping, strict data validation, and zero-leakage split preservation (Train: 10,269, Valid: 1,284, Test: 1,283).
- **Phase 2: Baseline Classification** ([`src/classification/`](file:///c:/Users/pk902/fake_news_detection/src/classification)): Ultra-fast CPU-only TF-IDF (10,000 unigrams + bigrams) with balanced Logistic Regression, Platt probability calibration, and persistence via Joblib.
- **Phase 3: Article & URL Ingestion** ([`src/ingestion/`](file:///c:/Users/pk902/fake_news_detection/src/ingestion)): Multi-format input handling (claims, headlines, raw articles, web URLs), readability extraction with `trafilatura` and `beautifulsoup4`, SSRF protection, length bounds, and error recovery.
- **Phase 4: Evidence Retrieval** ([`src/retrieval/`](file:///c:/Users/pk902/fake_news_detection/src/retrieval)): Curated institutional reference corpus, SentenceTransformers embeddings (`all-MiniLM-L6-v2`), FAISS vector index, and supporting vs. contradicting passage categorization.
- **Phase 5: LLM Verification** ([`src/verification/`](file:///c:/Users/pk902/fake_news_detection/src/verification)): Open-weight model verification (`google/flan-t5-small`), structured JSON output, explicit contradiction vs. missing evidence separation, and multi-factor uncertainty estimation.
- **Phase 6: Hybrid Pipeline & Dashboard** ([`frontend/`](file:///c:/Users/pk902/fake_news_detection/frontend), [`app.py`](file:///c:/Users/pk902/fake_news_detection/app.py)): Multi-model consensus engine (`EVIDENCE_OVERRIDE`, `STRONG_AGREEMENT`, `UNGROUNDED_HEURISTIC`), decision policy gatekeeping, and modern glassmorphic Streamlit UI.
- **Phase 7: Evaluation & Finalization** ([`reports/`](file:///c:/Users/pk902/fake_news_detection/reports), [`scripts/run_evaluation.py`](file:///c:/Users/pk902/fake_news_detection/scripts/run_evaluation.py)): 3-way comparative benchmark (Baseline vs. LLM-only vs. Hybrid), confidently incorrect post-mortem, figure generation, and formal capstone project report.

---

## Quickstart

### 1. Launch the Streamlit Web Dashboard (Phase 6)
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
*Open your browser at `http://localhost:8501` to use the interactive interface.*

### 2. Run the Full Test Suite (88 Unit & Integration Tests)
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### 3. Run End-to-End Analysis via CLI

#### Analyze with Mock LLM (Fast, deterministic)
```powershell
.\.venv\Scripts\python.exe src/pipeline/analysis_pipeline.py "Vaccines cause autism" --llm-backend mock
```

#### Analyze with Local Open-Weight Model (CPU execution)
```powershell
.\.venv\Scripts\python.exe src/pipeline/analysis_pipeline.py "NASA rover discovers organic molecules on Mars" --llm-backend local
```

#### Analyze a News URL with SSRF Protection
```powershell
.\.venv\Scripts\python.exe src/pipeline/analysis_pipeline.py "https://www.cdc.gov/vaccinesafety/concerns/autism.html" --llm-backend mock
```

---

## Component Architecture

### Phase 4: Evidence Retrieval
| Component | File | Description |
| :--- | :--- | :--- |
| **Reference Corpus** | [`data/evidence/reference_documents.json`](file:///c:/Users/pk902/fake_news_detection/data/evidence/reference_documents.json) | High-trust institutional knowledge from WHO, CDC, NASA, IPCC, NOAA, and Nature |
| **Passage Chunking** | [`src/preprocessing/chunking.py`](file:///c:/Users/pk902/fake_news_detection/src/preprocessing/chunking.py) | Sentence-boundary aligned passage chunking with contextual overlap |
| **Embeddings** | [`src/retrieval/embeddings.py`](file:///c:/Users/pk902/fake_news_detection/src/retrieval/embeddings.py) | Local CPU `all-MiniLM-L6-v2` dense embeddings (384-dim, normalized) |
| **FAISS Vector Store** | [`src/retrieval/vector_store.py`](file:///c:/Users/pk902/fake_news_detection/src/retrieval/vector_store.py) | Inner Product / Cosine similarity search with disk persistence |
| **Stance Analysis** | [`src/retrieval/search.py`](file:///c:/Users/pk902/fake_news_detection/src/retrieval/search.py) | Categorizes evidence into `SUPPORTS`, `CONTRADICTS`, or `NEUTRAL` |
| **Source Credibility** | [`src/retrieval/source_validation.py`](file:///c:/Users/pk902/fake_news_detection/src/retrieval/source_validation.py) | Ranks sources by authoritative credibility tiers (Tier 1 to Tier 3) |

### Phase 5: LLM Verification
| Component | File | Description |
| :--- | :--- | :--- |
| **Prompt Engineering** | [`src/verification/prompt_templates.py`](file:///c:/Users/pk902/fake_news_detection/src/verification/prompt_templates.py) | Structured evidence block formatting and JSON schema enforcement |
| **Model Loader** | [`src/verification/llm_loader.py`](file:///c:/Users/pk902/fake_news_detection/src/verification/llm_loader.py) | Supports `mock`, local `google/flan-t5-small`, and HuggingFace API backends |
| **Contradiction Detector** | [`src/verification/contradiction_detector.py`](file:///c:/Users/pk902/fake_news_detection/src/verification/contradiction_detector.py) | Key requirement: reliably separates contradictory evidence from missing evidence |
| **Evidence Quality** | [`src/verification/evidence_analyzer.py`](file:///c:/Users/pk902/fake_news_detection/src/verification/evidence_analyzer.py) | Measures passage relevance, coverage depth, and knowledge base gaps |
| **Uncertainty Estimator** | [`src/verification/uncertainty.py`](file:///c:/Users/pk902/fake_news_detection/src/verification/uncertainty.py) | Multi-factor uncertainty combining evidence (45%), model (25%), and alignment (30%) |
| **Decision Policy** | [`src/verification/decision_policy.py`](file:///c:/Users/pk902/fake_news_detection/src/verification/decision_policy.py) | 4-rule gatekeeper overriding ungrounded verdicts and flagging review |

### Phase 6: Hybrid Fusion & Web Application
| Component | File | Description |
| :--- | :--- | :--- |
| **Result Builder** | [`src/pipeline/result_builder.py`](file:///c:/Users/pk902/fake_news_detection/src/pipeline/result_builder.py) | Multi-model consensus synthesis (`STRONG_AGREEMENT`, `EVIDENCE_OVERRIDE`, etc.) |
| **Streamlit Entrypoint** | [`app.py`](file:///c:/Users/pk902/fake_news_detection/app.py) | Main server launcher (`streamlit run app.py`) |
| **Web Dashboard** | [`frontend/dashboard.py`](file:///c:/Users/pk902/fake_news_detection/frontend/dashboard.py) | Glassmorphic UI with tabs for claims, full articles, URLs, and JSON audit export |
| **UI Components** | [`frontend/components.py`](file:///c:/Users/pk902/fake_news_detection/frontend/components.py) | Glowing verdict hero cards, clickable evidence source links, and reasoning cards |
| **Visualizations** | [`frontend/visualizations.py`](file:///c:/Users/pk902/fake_news_detection/frontend/visualizations.py) | 3-way consensus breakdown matrix and calibrated confidence progress meters |

---

## Hybrid Consensus Logic

When evaluating news and claims, surface-level linguistic features (e.g. formal phrasing, grammar) often conflict with factual accuracy. Veritas AI handles these tensions through explicit consensus policies:

1. **Strong Agreement:** Both the statistical classifier and retrieved evidence agree $\rightarrow$ High confidence `TRUE` or `FALSE`.
2. **Evidence Override:** When a false claim uses sophisticated or deceptive phrasing that tricks a statistical classifier into predicting "Real", authoritative refuting passages override the surface style $\rightarrow$ Verdict is `FALSE` with `EVIDENCE_OVERRIDE` consensus.
3. **Missing Evidence:** When no authoritative evidence exists in the verified database $\rightarrow$ Verdict is overridden to `UNVERIFIABLE` (`UNGROUNDED_HEURISTIC`), preventing hallucinated verdicts.
4. **Conflict & Uncertainty:** When signals diverge or evidence is mixed $\rightarrow$ Verdict is set to `NEEDS_REVIEW` and flagged for human evaluation.

---

## Phase 7: Evaluation & Comparative Results

A dedicated evaluation script ([`scripts/run_evaluation.py`](file:///c:/Users/pk902/fake_news_detection/scripts/run_evaluation.py)) compares all three paradigms side-by-side:

| Metric | 1. Baseline (TF-IDF) | 2. LLM-Only (Zero-Shot) | 3. Hybrid Veritas AI |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 75.0% | 0.0% (all ungrounded) | **58.3%** |
| **Precision** | 85.7% | 0.0% | **71.4%** |
| **Recall (Fake Detection)** | 75.0% | 0.0% | **100.0%** |
| **F1-Score** | 80.0% | 0.0% | **83.3%** |
| **Confidently Incorrect Errors** | 1 (Linguistic Trap) | 0 | **1 (Safely Calibrated)** |
| **Gatekeeping 'UNVERIFIABLE'** | 0.0% (Forced Guessing) | 100.0% (Blind Uncertainty) | **25.0% (Correct Gatekeeping)** |
| **Evidence Grounding** | None | None | **100% Institutional Citations** |

Run the evaluation benchmark:
```powershell
.\.venv\Scripts\python.exe scripts/run_evaluation.py
```

### Comprehensive Project Reports & Documentation
- 📄 **[Final Project Report](file:///c:/Users/pk902/fake_news_detection/reports/project_report.md):** Complete formal capstone report covering problem statement, architecture, empirical findings, and ethical considerations.
- 📊 **[Evaluation Results](file:///c:/Users/pk902/fake_news_detection/reports/evaluation_results.md):** In-depth benchmark analysis and post-mortem on confidently incorrect predictions.
- 🔬 **[System Methodology](file:///c:/Users/pk902/fake_news_detection/reports/methodology.md):** Mathematical formulation, uncertainty calculations, and algorithmic breakdown across all 7 phases.
- ⚠️ **[System Limitations](file:///c:/Users/pk902/fake_news_detection/reports/limitations.md):** Transparent assessment of corpus boundaries, computational constraints, and failure modes.

---

## Repository Structure

```
fake_news_detection/
├── app.py                         # Streamlit application entry point
├── frontend/                      # Streamlit dashboard and UI components
│   ├── components.py              # Glassmorphic cards, evidence pills, hero verdicts
│   ├── dashboard.py               # Main multi-tab Streamlit dashboard
│   └── visualizations.py          # Consensus matrix and confidence gauges
├── src/
│   ├── classification/            # Phase 2: TF-IDF + Logistic Regression
│   ├── data/                      # Phase 1: LIAR dataset loader & validator
│   ├── ingestion/                 # Phase 3: URL, article, & claim extraction
│   ├── pipeline/                  # Phase 6: Hybrid pipeline & result builder
│   │   ├── analysis_pipeline.py   # End-to-end multi-stage pipeline
│   │   └── result_builder.py      # Consensus engine & decision fusion
│   ├── preprocessing/             # Text cleaning, normalization, chunking
│   ├── retrieval/                 # Phase 4: FAISS vector index & embeddings
│   ├── schemas/                   # Pydantic data contracts (evidence, requests, responses)
│   ├── utils/                     # Security, SSRF defense, error handling
│   └── verification/              # Phase 5: LLM verification & uncertainty
├── artifacts/
│   ├── models/baseline/           # Serialized baseline classifier & vectorizer
│   └── vector_store/              # Serialized FAISS index & passage metadata
├── data/
│   ├── raw/                       # LIAR benchmark splits
│   └── evidence/                  # Authoritative reference corpus
└── tests/                         # Comprehensive pytest test suite (88 tests)
```
