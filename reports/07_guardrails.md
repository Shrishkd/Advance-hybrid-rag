# Phase 7 — Guardrails

## Scope: similarity to the corpus

Calibrated threshold: **0.294** (min golden score 0.314 minus a 0.02 margin).

| set | n | min | median | max | blocked at threshold |
|---|---:|---:|---:|---:|---:|
| golden (in scope) | 50 | 0.314 | 0.537 | 0.646 | 0/50 |
| synthetic (in scope) | 250 | 0.198 | 0.586 | 0.817 | 2/250 |
| off-topic chat | 20 | 0.112 | 0.252 | 0.342 | 15/20 |
| pronoun follow-ups | 15 | 0.265 | 0.337 | 0.453 | 2/15 |

- **Golden false-block rate: 0/50** (0 by construction).
- **Synthetic false-block rate: 2/250** - held-out in-scope questions the threshold was NOT fitted on.
- **Off-topic caught: 15/20**. Any that pass are declined downstream by the NOT_IN_SOURCES rule.
- **Pronoun follow-ups that WOULD be blocked if scope ran mid-conversation: 2/15**. Most clear the bar because even 'Why?' shares register with textbook prose, but the margin is thin and short ones fall below it at random. So scope applies only on a session's first turn; later turns are condensed before retrieval.

## Prompt injection: pattern match

- Attacks caught: **10/10**
- False positives on 300 real questions: **0**

## Known gaps

- **Toxicity**: the specialist classifier (llama-guard3:1b) could not be downloaded (IPv6 failure). Not covered.
- **PII**: regex (email / phone / card), not Presidio NER - names and addresses are not detected.
- **Injection**: patterns catch known phrasings, not novel paraphrases. The real defence is structural: the generator only ever sees retrieved passages plus the question, and every citation is verified.
