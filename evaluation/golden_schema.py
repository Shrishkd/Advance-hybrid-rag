"""Schema for the golden evaluation set — Phase 3.

WHY GROUND-TRUTH CONTEXTS, NOT JUST ANSWERS
-------------------------------------------
Most RAG projects record `question -> expected_answer` and stop. That is only
enough to measure the system END TO END, which means every failure looks the
same: "wrong answer". You can never tell whether the retriever failed to find
the passage or the generator failed to use it — and those demand completely
different fixes.

Recording WHERE the answer lives makes retrieval measurable on its own:
recall@k, precision@k, MRR, nDCG, none of which need an LLM judge. That is the
whole reason Phase 4 can run hundreds of retrieval experiments cheaply.

Labelling contexts is tedious. It is also the single highest-value hour in the
project.

WHY PHYSICAL (PDF) PAGES
------------------------
`ContextRef` records PHYSICAL page numbers — the page your PDF reader shows —
never printed page numbers. Three reasons:

  1. It is what you actually see while labelling.
  2. Two books (Huyen, Jurafsky) have NO trustworthy printed-page offset, so
     printed numbers do not exist for them. Physical pages always do.
  3. It is CHUNKER-INDEPENDENT. Chunk IDs change with every chunking strategy;
     page ranges do not. One golden set stays valid across every Phase 4
     experiment. Label by chunk ID and the labels die the first time you
     change chunk_size.

A retrieved chunk counts as a hit when its [page_start, page_end] OVERLAPS a
labelled range. Overlap, not containment — an answer spanning a page boundary
must not be scored as a miss.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Book identifiers. MUST match `short` in configs/corpus.yaml — these strings
# are the join key between labels and chunks, so a rename invalidates the set.
BOOKS = {"Raschka", "Goodfellow", "Huyen", "Géron", "Bishop", "Jurafsky"}

QUESTION_TYPES = {
    "factual",          # a specific stated fact
    "definition",       # "what is X"
    "explanation",      # "why/how does X work"
    "comparison",       # X vs Y
    "multi_hop",        # needs 2+ passages chained
    "synthesis",        # combine several passages into something new
    "specific_source",  # "what does Bishop say about X" - tests metadata filtering
    "ambiguous",        # underspecified; tests clarification vs. guessing
    "numerical",        # a number, formula, or quantity
    "unanswerable",     # NOT in the corpus; tests refusal
}

DIFFICULTIES = {"easy", "medium", "hard", "unanswerable"}

# What the SYSTEM should do — distinct from whether the evidence exists.
#
# `answerable` and `expected_behaviour` were originally one flag, and that
# conflated two genuinely different things:
#
#   answerable          = is the evidence in the corpus?   (RETRIEVAL question)
#   expected_behaviour  = what should the system do?       (GENERATION question)
#
# An ambiguous question breaks the conflation. "What does 'make' mean in 'I
# made her duck'?" IS in the corpus — Jurafsky p4 discusses five readings — so
# retrieval must find that page and is scored on it. But the right response is
# to enumerate the readings or ask which was meant, NOT to pick one. Evidence
# present, direct answer wrong.
#
# Collapsing that into one boolean forces a choice between two falsehoods:
# mark it unanswerable and retrieval can never be scored on it, or mark it
# answerable and the generator is graded as if guessing were correct.
BEHAVIOURS = {
    "answer",    # evidence exists, give the answer
    "clarify",   # evidence exists, but the question is underspecified
    "refuse",    # no evidence in corpus; answering at all is the failure
}


@dataclass
class ContextRef:
    """Where an answer lives. Physical PDF pages, inclusive on both ends."""
    book: str
    page_start: int
    page_end: int

    def overlaps(self, other_start: int, other_end: int) -> bool:
        """Standard interval overlap.

        >>> ContextRef("Bishop", 100, 105).overlaps(104, 110)
        True
        >>> ContextRef("Bishop", 100, 105).overlaps(106, 110)
        False
        """
        return self.page_start <= other_end and other_start <= self.page_end


@dataclass
class GoldenItem:
    """One benchmark question.

    `ground_truth_contexts` is EMPTY for unanswerable items — that is the point
    of them. A system that retrieves something anyway and answers confidently
    is failing in the most damaging way a RAG system can, because the output
    looks exactly like a correct one.
    """
    qid: str
    question: str
    question_type: str
    difficulty: str
    answerable: bool
    ground_truth_answer: str | None
    ground_truth_contexts: list[ContextRef] = field(default_factory=list)
    notes: str = ""
    expected_behaviour: str = "answer"

    @property
    def cross_document(self) -> bool:
        """True when answering requires more than one book.

        Derived, not stored: storing it invites it drifting out of sync with
        the contexts it describes.
        """
        return len({c.book for c in self.ground_truth_contexts}) > 1

    def validate(self) -> list[str]:
        """Return problems with this item. Empty list means valid."""
        errs: list[str] = []
        if self.question_type not in QUESTION_TYPES:
            errs.append(f"{self.qid}: bad question_type {self.question_type!r}")
        if self.difficulty not in DIFFICULTIES:
            errs.append(f"{self.qid}: bad difficulty {self.difficulty!r}")
        for c in self.ground_truth_contexts:
            if c.book not in BOOKS:
                errs.append(f"{self.qid}: unknown book {c.book!r}")
            if c.page_start > c.page_end:
                errs.append(f"{self.qid}: inverted page range {c}")

        if self.expected_behaviour not in BEHAVIOURS:
            errs.append(
                f"{self.qid}: bad expected_behaviour {self.expected_behaviour!r}"
            )

        # Coherence rules. These key off `answerable`, which now means exactly
        # one thing: is the evidence in the corpus.
        if self.answerable and not self.ground_truth_contexts:
            errs.append(
                f"{self.qid}: answerable but no ground-truth contexts — "
                "retrieval cannot be scored on this item"
            )
        if not self.answerable and self.ground_truth_contexts:
            errs.append(
                f"{self.qid}: not answerable from the corpus but has contexts — "
                "contradictory. An ambiguous question whose discussion IS in the "
                "corpus should be answerable=true, expected_behaviour='clarify'."
            )
        if self.answerable and not self.ground_truth_answer:
            errs.append(f"{self.qid}: answerable but no ground_truth_answer")

        # Behaviour must agree with evidence.
        if self.expected_behaviour == "refuse" and self.answerable:
            errs.append(f"{self.qid}: expects refusal but is answerable")
        if self.expected_behaviour in ("answer", "clarify") and not self.answerable:
            errs.append(
                f"{self.qid}: expects {self.expected_behaviour!r} but is not answerable"
            )
        if self.question_type == "unanswerable" and self.expected_behaviour != "refuse":
            errs.append(f"{self.qid}: type 'unanswerable' must expect 'refuse'")
        if self.question_type == "ambiguous" and self.expected_behaviour != "clarify":
            errs.append(f"{self.qid}: type 'ambiguous' must expect 'clarify'")
        return errs


# ───────────────────────────────────────────────────────────────────────
# Persistence
# ───────────────────────────────────────────────────────────────────────

def load(path: Path) -> list[GoldenItem]:
    """Load a golden set from JSONL."""
    items: list[GoldenItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        # Curation provenance (_source_excerpt, _line, ...) and the review
        # page's `keep` flag are kept in the file deliberately - they are how
        # you audit a label months later. They are not part of the schema, so
        # drop them here rather than forcing the curated file to be stripped.
        d = {k: v for k, v in d.items() if not k.startswith("_") and k != "keep"}
        d["ground_truth_contexts"] = [
            ContextRef(**c) for c in d.get("ground_truth_contexts", [])
        ]
        items.append(GoldenItem(**d))
    return items


def save(path: Path, items: list[GoldenItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def report(items: list[GoldenItem]) -> str:
    """Coverage summary — surfaces thin cells before they distort results.

    With a 50-item cap across 10 types, a cell holding one question is a cell
    whose 'result' is a coin flip. Better to see that while the set is still
    being built than to discover it while reading a benchmark table.
    """
    from collections import Counter

    by_type = Counter(i.question_type for i in items)
    by_diff = Counter(i.difficulty for i in items)
    by_book = Counter(c.book for i in items for c in i.ground_truth_contexts)

    by_beh = Counter(i.expected_behaviour for i in items)
    lines = [
        f"total: {len(items)}  cross-document: {sum(i.cross_document for i in items)}",
        f"unanswerable: {sum(not i.answerable for i in items)}",
        "behaviour:     " + ", ".join(f"{k}={v}" for k, v in sorted(by_beh.items())),
        "",
        "by type:       " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())),
        "by difficulty: " + ", ".join(f"{k}={v}" for k, v in sorted(by_diff.items())),
        "by book:       " + ", ".join(f"{k}={v}" for k, v in sorted(by_book.items())),
    ]
    missing = QUESTION_TYPES - set(by_type)
    if missing:
        lines.append(f"MISSING TYPES: {sorted(missing)}")
    thin = [k for k, v in by_type.items() if v < 3]
    if thin:
        lines.append(f"THIN (<3 items, results will be noise): {sorted(thin)}")
    return "\n".join(lines)
