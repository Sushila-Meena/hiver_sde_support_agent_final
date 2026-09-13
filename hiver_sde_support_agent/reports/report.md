# Hiver SDE Intern — AppleSupport AI Support Agent

> **Important:** All numeric result fields below must be filled from actual runs. Never invent metrics or human labels.

## 1. Problem framing

The goal is to build a support agent for one brand that (1) classifies an incoming customer message into a small, data-derived intent set, (2) drafts a response grounded in historically observed resolutions, and (3) decides whether to auto-handle or escalate, with a reason.

**Brand:** AppleSupport

**Good means:** high intent macro-F1, useful historical retrieval, grounded replies, conservative escalation on uncertain/high-risk cases, and reproducible evaluation on a held-out human-labelled set.

**Not built:** account lookup, password resets, refunds, private verification, direct actions in Apple systems, or any workflow requiring access to private customer data.

## 2. Data and split

Source: Customer Support on Twitter (TWCS). The source provides tweet IDs, author IDs, inbound/outbound direction, timestamps, text, and reply-link fields that allow conversation reconstruction.

We reconstruct customer → AppleSupport response pairs using `author_id == AppleSupport` and `in_response_to_tweet_id`.

**Leakage control:** sort pairs by timestamp. The first 80% is development data; the final 20% is held out. The retrieval index contains only development data, and the golden examples are sampled only from the held-out period.

Training labels are manually assigned to a separate development candidate file after intent discovery. The golden set is never used for training or retrieval.

## 3. System

```text
                 ┌──────────────────────┐
Customer message │ normalize + classify │
        ────────►│ TF-IDF + LogisticReg │
                 └──────────┬───────────┘
                            │ intent/confidence
                            ▼
                 ┌──────────────────────┐
                 │ historical retrieval │
                 │ TF-IDF cosine search │
                 └──────────┬───────────┘
                            │ evidence
                            ▼
                 ┌──────────────────────┐
                 │ grounded reply draft │
                 │ LLM / safe fallback  │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ deterministic router │
                 │ AUTO / HUMAN + reason│
                 └──────────────────────┘
```

The LLM is not allowed to invent account-specific actions, policies, refunds, timelines, or guarantees. Routing is outside the LLM so safety and uncertainty do not depend on a single generation.

## 4. Results vs baselines

| System | Intent Acc | Intent Macro-F1 | Routing Acc | Reply Quality |
|---|---:|---:|---:|---:|
| Majority-intent baseline | `[FILL]` | `[FILL]` | — | — |
| Simple lexical baseline | `[FILL]` | `[FILL]` | — | — |
| Proposed classifier + retrieval + grounded LLM | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |

For retrieval, report the mean top-1 similarity and a manually defined retrieval-quality measure if the golden set contains evidence labels. Do **not** call “the query exists in the index” R@5 because that would be leakage.

## 5. LLM-as-judge validity

Calibration set: `[FILL]` human-labelled examples.

- Pearson correlation: `[FILL]`
- Weighted Cohen's kappa: `[FILL]`
- Mean absolute error: `[FILL]`

The judge should only be used as a reported metric after this calibration step. Human labels remain the reference.

## 6. Top 5 failures

For each failure include the actual customer message, prediction, expected label/decision, retrieved evidence, and a hypothesis.

1. `[REAL EXAMPLE + HYPOTHESIS]`
2. `[REAL EXAMPLE + HYPOTHESIS]`
3. `[REAL EXAMPLE + HYPOTHESIS]`
4. `[REAL EXAMPLE + HYPOTHESIS]`
5. `[REAL EXAMPLE + HYPOTHESIS]`

## 7. What is misleading about my headline number?

A single average accuracy can hide class imbalance, rare-intent failures, confidence calibration problems, retrieval mistakes, and unsafe errors. A high reply-quality score can also be misleading if the model is merely copying a nearby historical response that is inappropriate for the current case. The evaluation therefore separates intent, retrieval, reply quality, and routing, and uses a time-held-out golden set.

## 8. What I would do with one more week

- Replace lexical retrieval with a compact embedding index and compare retrieval recall.
- Add conversation context rather than classifying a single tweet in isolation.
- Use active learning to label uncertain/novel examples first.
- Calibrate the auto-handle threshold against the cost of unsafe automation.
- Expand judge calibration and adjudication with a second human reviewer.
- Add a small reviewer UI showing intent, confidence, evidence, draft, and escalation reason.
- Add monitoring for intent drift and retrieval-quality drift.
