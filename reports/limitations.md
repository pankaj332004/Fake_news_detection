# Veritas AI — System Limitations & Failure Modes

A transparent assessment of the technical boundaries, failure modes, and operational constraints of the Veritas AI platform.

---

## 1. Reference Corpus Boundaries & Knowledge Horizons

- **The Closed-World Assumption:** Veritas AI's fact verification accuracy is strictly bounded by the coverage and freshness of its indexed reference documents (`data/evidence/reference_documents.json`).
- **Unindexed Domains:** If a claim concerns emerging news, local political scandals, or niche technical subjects not yet incorporated into the vector index, the system correctly defaults to `UNVERIFIABLE`. While this is safe and prevents hallucinations, it requires continuous automated index ingestion in production settings.
- **Dynamic Truths:** Facts change over time (e.g., elected officials, economic statistics, scientific discoveries). Static vector stores require automated update pipelines with temporal metadata tagging.

---

## 2. Statistical Baseline Vulnerabilities

- **Grammar & Prose Bias:** TF-IDF classifiers evaluate vocabulary frequency rather than factual veracity. Consequently, sophisticated disinformation crafted in academic, formal, or official-sounding language frequently tricks the baseline classifier into predicting "Real".
- **Short Texts & Lack of Context:** Claims under 10 words often lack sufficient n-gram density to establish strong statistical priors, resulting in diffuse probability distributions near 50%.
- **Sarcasm & Satire:** Sarcastic headlines and satire (e.g., *The Onion*) rely on pragmatic context that simple n-gram feature spaces fail to capture.

---

## 3. Computational & Model Tradeoffs

- **CPU vs. GPU Footprint:** To ensure Veritas AI runs on consumer hardware without dedicated GPU clusters, the system utilizes lightweight open-weight models (`google/flan-t5-small`, ~300MB; `all-MiniLM-L6-v2`, ~90MB). While these models execute with zero cost on CPU, their contextual reasoning capacity is naturally lower than massive 70B+ parameter models.
- **Context Length Limitations:** Small transformer models are limited to 512 input tokens. When evaluating long-form news articles, Veritas AI deconstructs the document into atomic propositions, which adds extraction overhead.

---

## 4. Stance Detection Challenges

- **Subtle Nuances:** Determining whether a passage corroborates or refutes a complex claim can be ambiguous when passages discuss related background information without directly addressing the claim's core assertion.
- **"Half-True" and Cherry-Picked Claims:** Misinformation that takes real statistics out of context is among the hardest to categorize automatically. Veritas AI addresses this by escalating such borderline cases to `NEEDS_REVIEW`.

---

## 5. Web Ingestion & Anti-Scraping Defenses

- **Paywalls & Cloudflare Protections:** Ingestion of live news URLs is vulnerable to JavaScript-rendered paywalls, CAPTCHAs, and bot-blocking services. When extraction is blocked, the system alerts the user and requests direct text pasting.
- **Boilerplate Noise:** While Trafilatura effectively strips navigation menus and advertisements, complex interactive web layouts occasionally introduce residual navigational text into the evaluation pipeline.
