# CogniStream — AI-Powered Clinical Trial Matching Engine

> 🏆 **Winner — COHERENCE Hackathon 2026** · Health Care & Medtech track · Team Paradigm
> Forked and maintained by [Dhruv Save](https://github.com/dhruvsave29) from the original [Paradigm-AlgoForge26](https://github.com/Abhishek222983101/Paradigm-AlgoForge26) repository.

CogniStream is an AI-powered clinical trial eligibility engine that reduces patient-to-trial matching from **30+ minutes of manual coordinator work to under 15 seconds**. It ingests a de-identified patient record, runs it through a 4-layer matching pipeline, and returns ranked, explainable trial matches — every pass/fail decision auditable.

---

## Table of Contents

- [The Problem](#the-problem)
- [How It Works](#how-it-works)
- [The 4-Layer Matching Pipeline](#the-4-layer-matching-pipeline)
- [Composite Scoring](#composite-scoring)
- [Privacy & HIPAA Safeguards](#privacy--hipaa-safeguards)
- [The ML Core](#the-ml-core)
- [Web Application](#web-application)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Running the ML Core](#running-the-ml-core)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Performance](#performance)
- [Team Paradigm](#team-paradigm)
- [License](#license)

---

## The Problem

Trial coordinators manually screen patient records against trial eligibility criteria — inclusion/exclusion text, biomarkers, ECOG status, lab values, staging, geography. A single patient can take **30+ minutes** across dozens of trials, and the process doesn't scale.

**CogniStream does it in under 15 seconds**, with per-criterion reasoning a human can verify.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  Patient Record (synthetic, de-identified)                      │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 · Secure Ingestion & Anonymization                     │
│  NER + regex PII redaction → AES-256 encryption → audit log     │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 · Deterministic Rule Engine (no ML)                    │
│  age · gender · ECOG · stage · lab values (Hb, WBC, platelets,  │
│  creatinine, bilirubin, ALT, AST) vs. parsed criteria           │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 · Semantic Matching (ML)                               │
│  PubMedBERT embeddings · cosine similarity · fine-tuned         │
│  Mistral-7B (QLoRA) eligibility reasoning with RAG              │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4 · Geographic Filtering & Ranking                       │
│  Google Maps Distance Matrix · trial-site radius scoring        │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Ranked matches · per-criterion pass/fail reasoning · score     │
│  breakdown · overall narrative explanation                      │
└─────────────────────────────────────────────────────────────────┘
```

## The 4-Layer Matching Pipeline

### Layer 1 — Secure Ingestion & Anonymization
`src/lib/hipaa.ts` + `ML_Core/engine/ner_anonymizer.py`
- Hybrid NER + pattern detection for names, emails, phones, addresses, dates, SSNs, Aadhaar and PAN numbers, medical record numbers, hospitals and physician names.
- Every PII entity is replaced with a deterministic, reversible-in-code pseudonym (`PATIENT_*`, `MRN-*`, `+91-XXXXX-*`) before it ever reaches an LLM.
- The full entity map, timestamp and method (`ner | pattern | hybrid`) are returned alongside the anonymized payload for auditability.

### Layer 2 — Deterministic Rule Engine
`ML_Core/engine/rule_engine.py`
- **Zero ML.** Pure logic evaluating hard criteria: age ranges, gender, ECOG performance status, disease stage and lab values (hemoglobin, WBC, platelets, creatinine, bilirubin, ALT, AST).
- Raw ClinicalTrials.gov inclusion/exclusion text is decomposed into structured `ParsedCriterion` objects (`ML_Core/engine/criteria_parser.py`) using clinical regex patterns, with an LLM fallback for unusual phrasing.
- A failed exclusion criterion applies a hard penalty (see scoring below) — no fuzzy semantics can "pass" a patient who is excluded.

### Layer 3 — Semantic Vector Search + LLM Reasoning
`ML_Core/engine/embedding_matcher.py`, `ML_Core/engine/llm_matcher.py`, `src/lib/ragEngine.ts`
- **Embeddings:** [S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO) sentence embeddings compute cosine similarity between the patient profile and trial criteria — vocabulary overlap like "EGFR-mutated NSCLC" vs. "lung carcinoma" resolves correctly.
- **Fine-tuned LLM:** Mistral-7B-Instruct-v0.3 + QLoRA adapter (`models/trialmatch-7b-lora`, see [The ML Core](#the-ml-core)) performs per-criterion eligibility reasoning; falls back to the Mistral API (`mistral-large-latest`) when no local model is available.
- **RAG:** trials and patients are chunked with metadata (phase, status, sponsor, conditions, locations, biomarkers, age/gender/ECOG constraints) into a retrieval index that grounds every LLM judgement in the actual record — no hallucinated criteria.

### Layer 4 — Geographic Filtering
`src/lib/googleDistance.ts` + `src/app/api/maps/distance/route.ts` (server-side proxy for the Google Maps APIs)
- Every trial location carries real coordinates for 10 Indian clinical-trial hubs (Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Pune, Kolkata, Ahmedabad, Vellore, Manipal).
- Distance-based scoring respects each trial's own recruitment radius (`radiusKm`), so site accessibility is part of the match quality, not an afterthought.

## Composite Scoring

```
composite = 0.30 × rule + 0.20 × embedding + 0.35 × llm + 0.15 × geo

if exclusion_failed: composite ×= 0.1   (hard penalty — safety first)
```

| Component | Weight | Source | Meaning |
|---|---|---|---|
| Rule | 0.30 | Deterministic engine | Hard clinical criteria met |
| Embedding | 0.20 | PubMedBERT | Semantic similarity of diagnosis vs. conditions |
| LLM | 0.35 | Fine-tuned Mistral-7B | Reasoned eligibility judgement |
| Geo | 0.15 | Google Maps | Travel burden to the nearest site |

Every result ships a full breakdown: per-criterion `pass | fail | warning` status, the patient's value vs. the rule's value, human-readable reasoning, and whether the criterion was decided by a rule or by ML (`isML`). A narrative overall explanation is generated from the score bands (`Excellent ≥ 80 · Good ≥ 60 · Partial ≥ 40 · Low < 40`).

## Privacy & HIPAA Safeguards

- **Anonymize before anything else** — no raw PII reaches any model or API.
- **AES-256 encryption at rest** (Node `crypto`, 32-byte key, per-record IV).
- **Append-only audit log** (`src/lib/auditLog.ts`): 13 typed actions (`CREATE_MATCH`, `EXPORT_DATA`, `ANONYMIZE_DATA`, `VIEW_PATIENT`, `FAILED_LOGIN`, `SESSION_TIMEOUT`, …) with user, role, resource and timestamp.
- **Synthetic-only data** — all 52 patients are machine-generated; no real PHI exists anywhere in this repo.

## The ML Core

`ML_Core/` is the Python training & inference side (FastAPI + PyTorch + HuggingFace):

| Component | File | What it does |
|---|---|---|
| Criteria parser | `engine/criteria_parser.py` | Regex → structured criteria, LLM fallback |
| Rule engine | `engine/rule_engine.py` | Deterministic hard-criteria evaluation |
| Embedding matcher | `engine/embedding_matcher.py` | PubMedBERT cosine similarity (CPU) |
| LLM matcher | `engine/llm_matcher.py` | Local QLoRA Mistral-7B → Mistral API fallback |
| Composite scorer | `engine/scorer.py` | Weighted fusion + exclusion penalty + ranking |
| Data fetcher | `scripts/fetch_trials.py` | Pulls real India trials from the ClinicalTrials.gov API v2 |
| Patient generator | `scripts/generate_patients.py` | Groq `llama-3.1-70b-versatile` generates realistic synthetic Indian patient records |
| Training-data builders | `scripts/generate_training_pairs.py`, `generate_training_hybrid.py` | Build ChatML training pairs from rule + engine outputs |
| Fine-tuner | `scripts/finetune.py` | QLoRA fine-tuning (see below) |

**Fine-tuning run (actual, logged in `ML_Core/training.log`):**

| Setting | Value |
|---|---|
| Base model | Mistral-7B-Instruct-v0.3 |
| Method | QLoRA 4-bit (fits 8 GB VRAM), LoRA r=16, α=32, dropout 0.05 |
| Data | 841 ChatML training pairs (`data/trialmatch_training.jsonl`) |
| Epochs | 3 · batch 2 × grad-accum 4 · LR 2e-4 cosine decay |
| Final train loss | **0.54** (final epoch run mean) |
| Token accuracy | **~93%** |
| Runtime | ~40 min on a single RTX GPU |
| Output | `models/trialmatch-7b-lora` adapter |

## Web Application

Next.js 14 (App Router) + React 18 + Tailwind CSS — a brutalist, coordinator-first UI:

| Route | Purpose |
|---|---|
| `/` | Landing — the 30 min → 15 s pitch |
| `/pipeline` | Watch the 4-layer pipeline process a patient step by step |
| `/results` | Ranked trial matches with score breakdowns and per-criterion reasoning |
| `/dashboard` | Coordinator dashboard — patient/trial browsing, audit visibility |
| `/chat` | RAG-grounded clinical assistant (Groq), floating chatbot on every page |

API routes (all server-side): `/api/hipaa/anonymize`, `/api/audit/log`, `/api/chat`, `/api/maps/distance`, `/api/maps/geocode`.

## Getting Started

### Prerequisites
- Node.js 18+ · npm
- Python 3.11+ (only for the ML Core / training)

### Installation

```bash
git clone https://github.com/dhruvsave29/cognistream.git
cd cognistream
npm install
```

### Environment Setup

```bash
cp .env.example .env.local
```

### Run the Web App

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) → start at `/pipeline` to watch a full match run.

## Environment Variables

| Variable | Used for | Where to get it |
|---|---|---|
| `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY` | Map rendering + distance matrix (enable *Maps JavaScript API* and *Distance Matrix API*) | [Google Cloud Console](https://console.cloud.google.com/) |
| `GROQ_API_KEY` | RAG chat assistant | [console.groq.com/keys](https://console.groq.com/keys) |
| `MISTRAL_API_KEY` | LLM reasoning fallback + complex criteria parsing | [console.mistral.ai](https://console.mistral.ai/) |
| `ENCRYPTION_KEY` | AES-256 record encryption (32 chars) | Generate: `openssl rand -hex 16` |

## Running the ML Core

```bash
cd ML_Core
pip install -r requirements.txt

# 1. Fetch real Indian trials from ClinicalTrials.gov
python scripts/fetch_trials.py

# 2. Generate synthetic patients (needs GROQ_API_KEY in ML_Core/.env)
python scripts/generate_patients.py

# 3. Build training pairs and fine-tune the QLoRA adapter
python scripts/generate_training_pairs.py
python scripts/finetune.py        # logs to Weights & Biases
```

The web app runs standalone with the bundled datasets — the ML Core is only needed for retraining or running the local Mistral-7B inference instead of the API.

## Project Structure

```
cognistream/
├── src/
│   ├── app/
│   │   ├── api/                  # anonymize · audit · chat · maps
│   │   ├── chat/                 # RAG assistant page
│   │   ├── dashboard/            # coordinator dashboard
│   │   ├── pipeline/             # 4-step live pipeline view
│   │   └── results/              # ranked matches + reasoning
│   ├── components/ui/            # FloatingChatbot · GoogleMaps · HIPAACompliance
│   └── lib/
│       ├── clinicalData.ts       # typed patient & trial models + datasets
│       ├── hipaa.ts              # anonymization + AES-256 encryption
│       ├── ragEngine.ts          # chunking + retrieval index
│       ├── compositeScoring.ts   # weighted scoring + explanations
│       ├── googleDistance.ts     # distance/geo scoring
│       └── auditLog.ts           # HIPAA audit trail
└── ML_Core/
    ├── engine/                   # criteria_parser · rule_engine · embedding_matcher
    │                             # llm_matcher · ner_anonymizer · scorer
    ├── scripts/                  # fetch_trials · generate_patients
    │                             # generate_training_* · finetune
    ├── data/                     # patients.json (52) · trials.json (97)
    │                             # trialmatch_training.jsonl (841 pairs)
    ├── requirements.txt
    └── training.log              # the actual QLoRA training run
```

## Datasets

| File | Count | Contents |
|---|---|---|
| `ML_Core/data/trials.json` | **97 trials** | Real Indian trials sourced from ClinicalTrials.gov — oncology (NSCLC, breast, colorectal…), diabetes, cardiovascular, TB — with full criteria, phases, sponsors and site coordinates |
| `ML_Core/data/patients.json` | **52 patients** | Synthetic Indian patient records generated by Groq Llama-3.1-70B: demographics, ICD-10 diagnoses, staging, biomarkers (EGFR, HER2…), ECOG, lab panels, medications, treatment history |
| `ML_Core/data/trialmatch_training.jsonl` | **841 pairs** | ChatML-formatted patient-trial eligibility reasoning pairs for QLoRA fine-tuning |

## Performance

| Metric | Before | After |
|---|---|---|
| Patient-to-trial screening (52 patients × 97 trials) | 30+ min manual per patient | **< 15 seconds automated** |
| Eligibility decisions | Coordinator judgement | Per-criterion machine reasoning, human-verifiable |
| PII exposure risk | Raw records in play | Anonymized + AES-256 + audit-logged |

Validated end-to-end on the bundled synthetic corpus — every match result is explainable down to the individual criterion.

## Team Paradigm

Built at the **IEEE AlgoForge'26 hackathon** by **Team Paradigm** — winner of the COHERENCE Hackathon 2026, Health Care & Medtech track.

Original repository: [Abhishek222983101/Paradigm-AlgoForge26](https://github.com/Abhishek222983101/Paradigm-AlgoForge26) · Maintained here by [Dhruv Save](https://github.com/dhruvsave29).

## License

MIT License — see usage terms of the original repository. Not a medical device; built as a hackathon prototype on synthetic data only.
