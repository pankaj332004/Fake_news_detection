# Veritas AI — System Methodology & Engineering Principles

This document details the engineering principles, algorithmic designs, mathematical formulations, and component integrations behind Veritas AI.

---

## 1. Dataset Hygiene & Leakage Prevention (Phase 1)

Veritas AI trains its baseline classifier on the **LIAR benchmark dataset** (Wang, ACL 2017), which comprises 12,836 human-labeled political and general statements from PolitiFact.
- **Split Preservation:** The splits defined by the benchmark creators are strictly preserved (Train: 10,269, Valid: 1,284, Test: 1,283).
- **Label Mapping:** PolitiFact's 6 fine-grained categories (`pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`) are mapped to binary ground truth (`pants-fire`, `false`, `barely-true` $\rightarrow$ `Fake` (1); `mostly-true`, `true` $\rightarrow$ `Real` (0)).
- **Leakage Prevention:** Vocabulary fitting and Platt scaling calibrators are fitted exclusively on the training split; validation and test splits are strictly held out.

---

## 2. Statistical Baseline Architecture (Phase 2)

The baseline classifier serves as a fast stylistic prior:
1. **Feature Extraction:** `TfidfVectorizer` with sublinear TF scaling ($1 + \log(\text{tf})$), n-gram range of $(1, 2)$, English stopword removal, and vocabulary capped at 10,000 top features.
2. **Classification Algorithm:** Balanced `LogisticRegression` with L2 regularization ($C=1.0$) and L-BFGS optimization:
   $$P(\text{Fake} \mid x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$
3. **Execution Profile:** Sub-millisecond CPU inference (< 5ms), enabling real-time operation without GPU requirements.

---

## 3. Secure Input Ingestion & Web Extraction (Phase 3)

The ingestion subsystem normalizes unstructured inputs into strongly typed Pydantic `ArticleDocument` contracts:
1. **Input Detection:** Automatically classifies inputs into `Claim` (short statement), `Headline` (single title line), `Article` (multiline text), or `URL` (web links).
2. **Text Normalization:** Unicode NFKC normalization, whitespace sanitization, HTML entity decoding, and control character stripping.
3. **SSRF & Security Defense:**
   - Validates URI scheme (whitelisting only `http` and `https`).
   - Resolves hostnames against private IP CIDR ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.0/8`, and link-local `169.254.0.0/16`).
   - Enforces 2,048-character URL length bounds and 10-second connect/read timeouts.
4. **Readability Extraction:** Primary parsing with `trafilatura` for clean main-body extraction; resilient fallback to `BeautifulSoup4` with automated noise stripping (`<script>`, `<style>`, `<nav>`, `<footer>`).

---

## 4. Dense Semantic Evidence Retrieval (Phase 4)

To ground veracity analysis in empirical reality, Veritas AI queries an indexed reference corpus of authoritative institutional knowledge:
1. **Reference Corpus:** Curated, peer-reviewed documents from the CDC, WHO, NASA, IPCC, NOAA, and Nature.
2. **Passage Chunking:** Text split on sentence boundaries with a target chunk size of 300 characters and 50-character contextual overlap, preserving semantic coherence.
3. **Dense Embeddings:** Local execution of `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense vectors with L2 normalization:
   $$\vec{v} = \frac{f_{\text{transformer}}(t)}{\|f_{\text{transformer}}(t)\|_2}$$
4. **FAISS Indexing:** Inner Product (`IndexFlatIP`) vector index matching normalized embeddings, equivalent to cosine similarity:
   $$\text{sim}(q, p) = \vec{v}_q \cdot \vec{v}_p \in [-1, 1]$$
5. **Stance Determination:** Lexical and semantic stance alignment categorizing each retrieved passage into `SUPPORTS`, `CONTRADICTS`, or `NEUTRAL`.
6. **Credibility Tiering:** Sources mapped to hierarchical credibility tiers (Tier 1: Governmental/Academic, Tier 2: Established News, Tier 3: General Media).

---

## 5. Open-Weight LLM Fact Verification (Phase 5)

Rather than using unconstrained prompt generation, Veritas AI enforces strict structured verification:
1. **Model Loader:** Supports local CPU execution of open-weight HuggingFace models (`google/flan-t5-small`, `flan-t5-base`) alongside deterministic mock and remote API backends.
2. **Prompt Template:** Injects retrieved evidence passages with unique IDs (`DOC-01#p0`), requires JSON output format, and mandates:
   - `verdict`: Raw categorical judgment (`TRUE`, `FALSE`, `UNVERIFIABLE`).
   - `reasoning`: Chain-of-thought analysis explaining evidence alignment.
   - `evidence_citations`: Exact passage IDs referenced.
   - `limitations`: Explicit statement of knowledge boundaries.
3. **Contradiction vs. Missing Evidence Distinction:**
   - Evaluates whether the reference documents directly refute the claim (`CONTRADICTED`) or whether the database simply lacks information on the topic (`ABSENT`).
   - Crucial distinction: Absence of evidence is not evidence of falsehood.

---

## 6. Multi-Factor Uncertainty & Decision Policy (Phase 6)

The hybrid decision policy acts as the supervisory gatekeeper:
1. **Uncertainty Formulation:**
   $$U = 0.45 \cdot (1 - Q_{\text{evidence}}) + 0.25 \cdot (1 - C_{\text{LLM}}) + 0.30 \cdot (1 - A_{\text{alignment}})$$
   Where $Q_{\text{evidence}}$ is evidence quality, $C_{\text{LLM}}$ is model confidence, and $A_{\text{alignment}}$ is consensus between statistical prior and evidence stance.
2. **Decision Policy Rules:**
   - **Rule 1 (Missing Evidence):** If evidence status is `ABSENT`, final verdict is strictly forced to `UNVERIFIABLE`.
   - **Rule 2 (High Uncertainty):** If uncertainty $U \ge 0.65$, flag for human review (`NEEDS_REVIEW`).
   - **Rule 3 (Evidence Override):** If statistical prior says "Real" but retrieved evidence contradicts the claim, evidence takes precedence $\rightarrow$ `FALSE` (`EVIDENCE_OVERRIDE`).
   - **Rule 4 (Aligned Consensus):** If evidence supports the claim and aligns with the LLM verdict $\rightarrow$ Adopt LLM verdict with calibrated hybrid confidence.

---

## 7. Streamlit Web Dashboard Architecture

The frontend is built using Vanilla Streamlit supplemented with custom glassmorphic CSS:
- **Zero Placeholder Guarantee:** Every card, link, and button connects directly to real backend functions.
- **Resource Caching:** `@st.cache_resource` loads the FAISS index and model weights into memory once on application startup, guaranteeing sub-second interactive response times.
- **Auditability:** Complete analysis payloads can be viewed in an interactive JSON viewer and downloaded as official compliance audit reports.
