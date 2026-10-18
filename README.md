# Fake News Detection System

An end-to-end fact-checking and veracity verification system built around the official **LIAR benchmark dataset** (Wang, ACL 2017). The system combines machine learning baseline classification, robust input ingestion, and automated verification.

---

## Architecture & Phases

- **Phase 1: Dataset Pipeline** ([`src/data/`](file:///c:/Users/pk902/fake_news_detection/src/data)): Official LIAR benchmark ingestion, 6-way and binary label mapping, strict data validation, and zero-leakage split preservation (Train: 10,269, Valid: 1,284, Test: 1,283).
- **Phase 2: Baseline Classification** ([`src/classification/`](file:///c:/Users/pk902/fake_news_detection/src/classification)): Ultra-fast CPU-only TF-IDF (10,000 unigrams + bigrams) with balanced Logistic Regression, Platt probability calibration, and persistence via Joblib.
- **Phase 3: Article & URL Ingestion** ([`src/ingestion/`](file:///c:/Users/pk902/fake_news_detection/src/ingestion)): Multi-format input handling (claims, headlines, raw articles, web URLs), readability extraction with `trafilatura` and `beautifulsoup4`, SSRF protection, length bounds, and error recovery.

---

## Quickstart

### 1. Run the Full Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### 2. Ingest and Prepare the LIAR Dataset
```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py
```

### 3. Train the Baseline Model
```powershell
.\.venv\Scripts\python.exe scripts/train_model.py --model-type baseline
```

### 4. Analyze Real User Inputs (Phase 3 Deliverable)

#### Analyze a Claim
```powershell
.\.venv\Scripts\python.exe scripts/analyze.py "NASA announced finding water on Mars."
```

#### Analyze a Raw Multiline Article
```powershell
.\.venv\Scripts\python.exe scripts/analyze.py "Breaking News`n`nScientists have verified a new superconductor material operating at room temperature."
```

#### Analyze a Web URL (with SSRF Safety & Boilerplate Extraction)
```powershell
.\.venv\Scripts\python.exe scripts/analyze.py "https://www.reuters.com/world/article-example"
```

---

## Input Ingestion Features (Phase 3)

| Feature | Implementation | Description |
| :--- | :--- | :--- |
| **Input Flexibility** | [`src/ingestion/document_loader.py`](file:///c:/Users/pk902/fake_news_detection/src/ingestion/document_loader.py) | Auto-detects and processes Claims, Headlines, Articles, and URLs |
| **Readable Text Extraction** | [`src/ingestion/html_parser.py`](file:///c:/Users/pk902/fake_news_detection/src/ingestion/html_parser.py) | High-precision boilerplate/ad removal via `trafilatura` + BeautifulSoup fallback |
| **SSRF Defense** | [`src/utils/security.py`](file:///c:/Users/pk902/fake_news_detection/src/utils/security.py) | Blocks loopback, private RFC1918 IPs, link-local addresses, and cloud metadata endpoints |
| **Input Guardrails** | [`src/utils/security.py`](file:///c:/Users/pk902/fake_news_detection/src/utils/security.py) | Enforces URL length bounds (max 2048 chars) and text bounds (5 to 50,000 chars) |
| **Failure Recovery** | [`src/utils/exceptions.py`](file:///c:/Users/pk902/fake_news_detection/src/utils/exceptions.py) | Gracefully handles HTTP 404, 403, timeouts, empty bodies, and invalid schemes |
