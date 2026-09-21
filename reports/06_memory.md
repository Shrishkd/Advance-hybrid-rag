# Phase 6 — Conversation memory

12 two-turn conversations. Turn 2 is a pronoun-only follow-up; its retrieval is scored against turn 1's labelled pages.

| condition | turn-2 recall@10 |
|---|---:|
| memory OFF (retrieve on the literal follow-up) | 0.083 |
| memory ON (condense, then retrieve) | 0.917 |

## What condensation produced

| turn 1 | follow-up | condensed | off | on |
|---|---|---|---:|---:|
| In Jurafsky's Levenshtein setup, what unit cost is assigned  | Why is that important? | Why is assigning a unit cost of 1 to an insertion operation important? | 0.00 | 1.00 |
| Which three categories of nodes are included in the Markov b | Can you explain it in simpler terms? | Can you explain the Markov blanket of a node \(x_i\) in simpler terms? | 0.00 | 1.00 |
| For the vocabulary size used in the passage, how many distin | What is the intuition behind it? | What is the intuition behind the number of distinct bigrams possible f | 0.00 | 0.00 |
| Which model's calibration curve is described as best because | How is that used in practice? | How is the calibration curve of the Logistic Regression model used in  | 0.00 | 1.00 |
| In the Berkeley study discussed by Huyen, how many creditwor | Why is that important? | Why is it important that 1.3 million creditworthy Black and Latino app | 1.00 | 1.00 |
| What macro-averaged F1 score does K-Nearest Neighbors achiev | Can you explain it in simpler terms? | Can you explain the macro‑averaged F1 score of 0.976410265560605 in si | 0.00 | 1.00 |
| During HMM training with EM, what sequence of E-step quantit | What is the intuition behind it? | What is the intuition behind the sequence of E‑step quantities and M‑s | 0.00 | 1.00 |
| How does Huyen define an edge case in an ML system, and what | How is that used in practice? | How is Huyen's definition of an edge case in an ML system used in prac | 0.00 | 1.00 |
| With drop_last enabled, how does the DataLoader handle a fin | Why is that important? | Why is discarding any final batch that contains fewer examples than th | 0.00 | 1.00 |
| What does Goodfellow mean by intractable inference, and why  | Can you explain it in simpler terms? | Can you explain Goodfellow's definition of intractable inference in si | 0.00 | 1.00 |
| In formal-grammar notation, what condition makes one string  | What is the intuition behind it? | What is the intuition behind the condition that makes one string direc | 0.00 | 1.00 |
| How does a decision-list classifier determine the sense of a | How is that used in practice? | How is the decision‑list classifier’s method of determining the sense  | 0.00 | 1.00 |
