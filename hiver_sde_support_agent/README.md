# Hiver SDE Intern — AppleSupport AI Support Agent

A reproducible implementation of the Hiver SDE take-home: **intent classification + historically grounded reply drafting + auto/human escalation**, with an evaluation-first design.

## What this project is optimizing for

The assignment explicitly says the proof is worth more than the system. This repo therefore separates:

- **development data** used for intent labelling, model training and retrieval;
- a **time-held-out golden set** of 150–250 hand-labelled examples;
- **two baselines**;
- automated intent/routing metrics;
- an **LLM-as-judge calibration** against human labels;
- real failure analysis and a decision log.

## Architecture

```text
Customer tweet
    │
    ├── TF-IDF + Logistic Regression ──► intent + confidence
    │
    ├── TF-IDF historical retrieval ──► brand-specific evidence
    │
    ├── grounded LLM draft ───────────► reply
    │
    └── deterministic safety router ───► AUTO / HUMAN + reason
```

## Dataset

Primary dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

Download with the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw
unzip data/raw/customer-support-on-twitter.zip -d data/raw/twcs
```

Expected file:

```text
data/raw/twcs/twcs.csv
```

The dataset's reply-link fields let us reconstruct customer → support responses. Do not commit the raw dataset to GitHub.

## Setup

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

For generated replies and judge scoring, configure Groq:

```powershell
$env:GROQ_API_KEY="YOUR_KEY"
$env:GROQ_MODEL="llama-3.3-70b-versatile"
```

The core retrieval/classification pipeline can still run without an LLM key; the fallback returns the strongest historical response and clearly marks that it is a fallback.

## Reproduce the pipeline

### 1. Build AppleSupport pairs

```bash
python scripts/build_brand_dataset.py
```

Output:

```text
data/processed/apple_support_pairs.csv
```

### 2. Discover the intent taxonomy

```bash
python scripts/discover_intents.py --k 12
```

Inspect:

```text
data/processed/intent_clusters.csv
```

Then update `configs/intent_taxonomy.yaml`. **Do not invent labels before looking at the real AppleSupport examples.**

### 3. Create development training-label candidates

```bash
python scripts/prepare_training_labels.py --n 500
```

Manually label `data/processed/train_labels.csv` using the taxonomy. This is separate from the golden evaluation set.

### 4. Train the intent model

```bash
python scripts/train_intent.py
```

The model is TF-IDF n-grams + Logistic Regression. A small transparent model is intentional: it is easy to explain and gives us a credible baseline.

### 5. Build the historical index

```bash
python scripts/build_index.py
```

**Important:** only the first 80% time slice enters the retrieval index. The golden set is never indexed.

### 6. Create the required golden evaluation set

```bash
python scripts/prepare_golden.py --n 200
```

Manually label:

```text
data/golden/golden_to_label.csv
```

Required fields:

- `gold_intent`
- `gold_auto_handle` = `yes` / `no`
- `gold_reply_quality` = 1–5
- `gold_notes`

A good sampling protocol is approximately 40% random, 30% intent-stratified, 20% hard/ambiguous, and 10% long/noisy/multi-intent examples. Record the actual protocol in the report.

### 7. Evaluate

```bash
python scripts/evaluate.py
```

This produces:

```text
reports/evaluation.json
reports/predictions.csv
```

Metrics include:

- majority-intent baseline accuracy/macro-F1;
- simple lexical baseline accuracy/macro-F1;
- proposed classifier accuracy/macro-F1;
- routing accuracy;
- mean top-1 retrieval similarity.

### 8. Run one live example

```bash
python scripts/run_agent.py --text "My iPhone update keeps failing"
```

The output contains:

- predicted intent;
- confidence;
- AUTO/HUMAN decision;
- explicit escalation reason;
- draft reply;
- retrieved historical evidence.

### 9. Failure analysis

```bash
python scripts/failure_analysis.py
```

Inspect `reports/top_intent_failures.csv` and turn the five most informative cases into the report's failure-analysis section.

## LLM-as-judge calibration

Generate a calibration set:

```bash
python scripts/calibrate_judge.py --n 30
```

Then a human must label `human_reply_quality` in:

```text
data/golden/judge_calibration.csv
```

Score judge-human agreement:

```bash
python scripts/score_judge_calibration.py
```

This reports Pearson correlation, weighted Cohen's kappa, and mean absolute error. The judge is **not** treated as ground truth.

## Baselines

### Baseline 1 — Majority intent

Always predicts the most frequent golden-set intent. This is the trivial baseline.

### Baseline 2 — Lexical intent

Matches the customer text against intent names/descriptions using token overlap. It is intentionally simple and transparent.

### Proposed system

TF-IDF + Logistic Regression intent classifier + historical retrieval + grounded LLM draft + deterministic escalation policy.

## Escalation policy

Escalate when:

- intent confidence is below the configured threshold;
- a high-risk security/legal/safety pattern is detected;
- the case requests a sensitive financial/account action;
- historical evidence is too weak.

The LLM never gets authority to perform an account action.

## Reproducibility and integrity

- Fixed random seeds are used for sampling.
- The raw Kaggle dataset is not committed.
- Golden examples are held out by time.
- Golden examples are excluded from retrieval.
- Human labels are never fabricated.
- Evaluation numbers are generated only after the pipeline is actually run.

## Submission mapping

| Hiver requirement | Repo location |
|---|---|
| Runnable pipeline | `scripts/` + `src/` |
| 150–250 hand-labelled golden set | `data/golden/golden_to_label.csv` |
| Automated evaluation | `scripts/evaluate.py` |
| LLM judge + human agreement | `scripts/calibrate_judge.py`, `scripts/score_judge_calibration.py` |
| Two baselines | `scripts/evaluate.py` |
| Failure analysis | `scripts/failure_analysis.py`, `reports/top_intent_failures.csv` |
| Six-page report | `reports/report.md` |
| Decision log | `reports/decision_log.md` |

## Sources

- Kaggle: Customer Support on Twitter — https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- Hiver assignment brief supplied with this repository.

Cite additional code/data/model sources if borrowed, as required by the assignment.
