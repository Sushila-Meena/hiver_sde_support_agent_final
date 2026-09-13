# Human labeling guide

## Intent labels

Use the taxonomy in `configs/intent_taxonomy.yaml`. Label the **customer's primary problem**, not the wording of the support response.

Rules:

1. Choose exactly one intent whenever possible.
2. If two issues are present, choose the issue that requires the main support action and note the secondary issue.
3. Use `unknown_other` only when none of the defined intents fits.
4. Do not create new labels in the CSV; propose taxonomy changes separately so the taxonomy stays stable during a labeling batch.
5. If a tweet is too vague to infer the problem, use `unknown_other` and explain why in `label_notes`.

## Auto-handle label

`yes` only when a support agent could safely provide a response using historical evidence without accessing the customer's account or taking a sensitive action.

Use `no` for cases involving:

- hacked/compromised accounts;
- fraud or unauthorized financial activity;
- legal/safety concerns;
- account-specific actions that require verification;
- ambiguous cases where the correct action is unclear;
- weak historical evidence.

## Reply quality (1–5)

- **5:** directly solves/addresses the issue, grounded, actionable, appropriate tone, no unsupported claims.
- **4:** useful and grounded, but slightly incomplete or less actionable.
- **3:** partially useful; important context/action is missing.
- **2:** mostly unhelpful or weakly grounded.
- **1:** wrong, unsafe, hallucinated, or clearly unrelated.

Use `gold_notes` for edge cases and disagreements.
