"""CRAG — grade the retrieved documents, re-retrieve once if they look weak.

WHAT PROBLEM IT TARGETS
-----------------------
The baseline trusts retrieval blindly. CRAG adds one check before generating: are
the top documents actually about the question? If none are, the question was
probably worded in a way retrieval could not match, so rewrite it and search once
more.

THE HARD PART IS THE THRESHOLD
------------------------------
`crag_route` is a graph routing condition. The real design question is choosing a
relevance threshold when the grader itself is noisy: trading needless re-retrieval
against answering from weak evidence. The reasoning is written out in the function
so it can be read, challenged and changed.

TWO LESSONS CARRIED FROM THE DECOMPOSITION RUNS
-----------------------------------------------
1. Small local models are unreliable at open-ended judgement - llama3.2:1b invented
   entities and swapped topics. So the grader here gets the narrowest possible task
   (yes/no on one short passage), and the router assumes it will be WRONG sometimes.
2. Replacing the user's question with a model's rewrite can only lose information.
   The rewrite is therefore FUSED with the original (RRF over both rankings), never
   swapped in, so a bad rewrite can add noise but cannot remove what the original
   query found.
"""

from __future__ import annotations

GRADE_PROMPT = (
    "You are checking search results for a question about machine learning.\n\n"
    "Question: {question}\n\n"
    "Passage: {passage}\n\n"
    "Does this passage contain information that helps answer the question? "
    "Reply with exactly one word: yes or no."
)

REWRITE_PROMPT = (
    "The search query below did not find useful passages in a library of machine "
    "learning textbooks. Rewrite it as ONE better search query: keep every "
    "technical term, remove filler words, and name the concept being asked about.\n\n"
    "Query: {question}\n\n"
    "Output only the rewritten query, on one line."
)

BATCH_GRADE_PROMPT = (
    "You are checking search results for a question about machine learning.\n\n"
    "Question: {question}\n\n"
    "{passages}\n\n"
    "For EACH numbered passage, does it contain information that helps answer the "
    "question? Reply with one line per passage, exactly like:\n"
    "1: yes\n2: no\n"
    "and nothing else."
)
"""ONE call grading all top passages.

WHY BATCHED AND IN THE CLOUD - measured, not preferred. The first CRAG run graded
each passage with a separate llama3.2:3b call. On this 7.4 GB machine that model
(~2 GB) could not coexist with the embedder and the served chatbot: free RAM fell to
0.2 GB, Windows compressed 1.9 GB of memory, and the run froze at 22/50 questions
along with the API. Local graders are preferred for cost; here a local grader
made the system unusable. One batched cloud call per question uses no local RAM and
replaces five calls with one.
"""


def parse_batch_grades(raw: str, n: int) -> list[bool]:
    """Parse '1: yes / 2: no' lines. Missing or garbled entries count as 'no'.

    >>> parse_batch_grades("1: yes\\n2: no\\n3: YES", 4)
    [True, False, True, False]
    >>> parse_batch_grades("garbage", 2)
    [False, False]
    """
    import re
    out = [False] * n
    for m in re.finditer(r"(\d+)\s*[:.)\-]\s*(yes|no)", raw, re.I):
        i = int(m.group(1)) - 1
        if 0 <= i < n:
            out[i] = m.group(2).lower() == "yes"
    return out


GRADE_TOP = 5          # documents graded per question
SNIPPET_CHARS = 400    # enough to judge topicality; short enough for a CPU model


def parse_grade(raw: str) -> bool:
    """Interpret a grader reply. Anything that is not clearly 'yes' is 'no'.

    Defaulting to 'no' on garbage is deliberate: an unparseable grade must not
    count as evidence that retrieval succeeded.

    >>> parse_grade("Yes."), parse_grade("no"), parse_grade("  YES, it does")
    (True, False, True)
    >>> parse_grade(""), parse_grade("maybe")
    (False, False)
    """
    return raw.strip().lower().startswith("yes")


def crag_route(grades: list[bool], attempt: int) -> str:
    """Decide whether retrieval was good enough to answer from.

    Args:
        grades: relevance verdicts for the top documents, best-ranked first.
        attempt: 0 on the first retrieval, 1 after one rewrite.

    Returns:
        "generate" or "rewrite".

    THE RULE, AND WHY EACH PART OF IT
    ---------------------------------
    Generate if AT LEAST ONE graded document is relevant; otherwise rewrite, once.

    - ONE is enough, not a majority. The generator reads the top 10 and needs one
      good passage to answer. Demanding more would trigger rewrites on questions
      whose answer lives in a single passage - most of the golden set.
    - A noisy grader pushes the same way. A small model will mark some relevant
      passages "no". With a high threshold those false negatives cause needless
      re-retrieval, each costing a rewrite call plus a second retrieval.
    - Rewrite ONCE, then answer regardless. The router never refuses. gpt-oss
      already errs toward refusing (wrong refusals were its main D9 weakness), and
      the answer prompt's NOT_IN_SOURCES rule is the right place to decline -
      it sees the passages, the router only sees yes/no flags.

    >>> crag_route([True, False, False, False, False], 0)
    'generate'
    >>> crag_route([False, False, False, False, False], 0)
    'rewrite'
    >>> crag_route([False, False, False, False, False], 1)
    'generate'
    """
    if any(grades):
        return "generate"
    return "rewrite" if attempt == 0 else "generate"
