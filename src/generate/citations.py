"""Citation parsing and verification — a hallucination detector that is free.

WHAT THIS CATCHES, AND WHY IT COSTS NOTHING
--------------------------------------------
The most damaging failure a RAG system has is a confident, fluent, well-cited
answer whose citation is invented. It looks exactly like a correct answer. A
reader who does not open the book cannot tell.

Because `context.assemble` hands the model explicitly numbered sources, that
failure becomes arithmetic:

    answer cites [S7], context held 5 sources  ->  FABRICATED, provably
    answer cites [S3]                          ->  resolves to a real chunk

No LLM judge, no cost, no rate limit, fully deterministic. Contrast Phase 8's
faithfulness metric, which asks a cloud model whether a claim is *supported* by
the context - a genuinely harder question that needs a judge. These two are
complementary:

    verifiable citations  ->  did it cite something real?      (here, free)
    faithfulness          ->  is the claim actually supported? (Phase 8, paid)

An answer can pass the first and fail the second - citing [S2] while saying
something [S2] does not support. It cannot fail the first and be trustworthy.

WHAT "UNCITED" MEANS AND WHY IT IS TRACKED SEPARATELY
------------------------------------------------------
An answer with no citations at all is not the same as one with a bad citation.
Both are problems, but they are different problems: the first is usually a
prompt-compliance failure (the model ignored the instruction), the second is a
fabrication. A generator that never cites will score perfectly on
"fabrication rate" while being useless, so the two are reported side by side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# [S1], [S12], and the common model habits: [S1, S2] and [S1][S2].
_CITE = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]", re.I)
_SID = re.compile(r"S\d+", re.I)


@dataclass
class CitationReport:
    """What an answer's citations did and did not do."""
    cited: list[str] = field(default_factory=list)       # unique, in order
    valid: list[str] = field(default_factory=list)       # resolve to a source
    fabricated: list[str] = field(default_factory=list)  # do not resolve
    n_citation_marks: int = 0                            # total, incl. repeats
    unused_sources: list[str] = field(default_factory=list)

    @property
    def has_citations(self) -> bool:
        return bool(self.cited)

    @property
    def fabrication_rate(self) -> float:
        """Fraction of distinct citations that point at nothing.

        Undefined with no citations, reported as 0.0. Always read alongside
        `has_citations` - an uncited answer trivially fabricates nothing.
        """
        return len(self.fabricated) / len(self.cited) if self.cited else 0.0

    @property
    def grounded(self) -> bool:
        """Cited at least once, and every citation resolves."""
        return self.has_citations and not self.fabricated

    def row(self) -> dict:
        return {
            "n_cited": len(self.cited),
            "n_marks": self.n_citation_marks,
            "n_fabricated": len(self.fabricated),
            "fabrication_rate": round(self.fabrication_rate, 4),
            "has_citations": self.has_citations,
            "grounded": self.grounded,
            "n_unused_sources": len(self.unused_sources),
        }


def extract_citations(answer: str) -> tuple[list[str], int]:
    """Pull source ids from an answer.

    Returns (unique ids in first-appearance order, total marks seen).
    Ids are upper-cased so [s3] and [S3] are the same source - a model
    lower-casing a tag is a formatting slip, not a different citation.

    >>> extract_citations("Regularization [S1] shrinks weights [S1], see [S3].")
    (['S1', 'S3'], 3)
    >>> extract_citations("Both apply [S1, S2].")
    (['S1', 'S2'], 2)
    >>> extract_citations("Adjacent [S1][S2] form.")
    (['S1', 'S2'], 2)
    >>> extract_citations("No citations here.")
    ([], 0)
    """
    ordered: list[str] = []
    total = 0
    for m in _CITE.finditer(answer):
        for sid in _SID.findall(m.group(1)):
            sid = sid.upper()
            total += 1
            if sid not in ordered:
                ordered.append(sid)
    return ordered, total


def verify(answer: str, available_sids: list[str]) -> CitationReport:
    """Check every citation against the sources actually supplied.

    Args:
        answer: the generated text.
        available_sids: ids that were in the prompt, e.g. ["S1","S2","S3"].

    >>> r = verify("Uses [S1] and [S4].", ["S1", "S2", "S3"])
    >>> r.valid, r.fabricated, r.grounded
    (['S1'], ['S4'], False)
    >>> round(r.fabrication_rate, 3)
    0.5
    >>> r.unused_sources
    ['S2', 'S3']
    >>> good = verify("Supported by [S2].", ["S1", "S2"])
    >>> good.grounded, good.fabrication_rate
    (True, 0.0)
    >>> silent = verify("An answer with no citation.", ["S1"])
    >>> silent.has_citations, silent.grounded, silent.fabrication_rate
    (False, False, 0.0)
    """
    have = {s.upper() for s in available_sids}
    cited, marks = extract_citations(answer)
    valid = [s for s in cited if s in have]
    fabricated = [s for s in cited if s not in have]
    unused = [s for s in available_sids if s.upper() not in set(cited)]
    return CitationReport(
        cited=cited,
        valid=valid,
        fabricated=fabricated,
        n_citation_marks=marks,
        unused_sources=unused,
    )


def expand(answer: str, sources: dict) -> str:
    """Replace [S<n>] with readable citations for display.

    Expansion uses the SOURCE metadata, never anything the model wrote, so a
    rendered citation cannot inherit a hallucination. Unresolvable ids are left
    verbatim and marked, so a fabrication stays visible in the output rather
    than being quietly dropped.

    >>> from src.generate.context import Source
    >>> src = {"S1": Source("S1", "Bishop", "...", 340, 341, 320, "printed",
    ...                     ["7. Sparse Kernel Machines"])}
    >>> expand("Margins matter [S1].", src)
    'Margins matter (Bishop, 7. Sparse Kernel Machines, p. 320).'
    >>> expand("Invented [S9].", src)
    'Invented [S9 — NO SUCH SOURCE].'
    """
    def repl(m: re.Match) -> str:
        names = []
        for sid in _SID.findall(m.group(1)):
            sid = sid.upper()
            s = sources.get(sid)
            names.append(s.human() if s else f"{sid} — NO SUCH SOURCE")
        if any("NO SUCH SOURCE" in n for n in names):
            return "[" + "; ".join(names) + "]"
        return "(" + "; ".join(names) + ")"

    return _CITE.sub(repl, answer)
